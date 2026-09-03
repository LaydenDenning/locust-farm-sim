"""Sensor-only anomaly classification for Phase 5."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.simulation.observations import Observation, Phase4Config, load_phase4_config


DETECTION_STATUSES = ("flagged", "clear", "unavailable")


class DetectionConfigError(ValueError):
    """Raised when Phase 5 configuration is invalid."""


@dataclass(frozen=True)
class DetectionRule:
    anomaly_threshold: float
    maximum_uncertainty: float
    allowed_quality_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        _unit_fraction(self.anomaly_threshold, "detection.anomaly_threshold")
        _unit_fraction(self.maximum_uncertainty, "detection.maximum_uncertainty")
        if not self.allowed_quality_flags:
            raise ValueError("detection.allowed_quality_flags must not be empty")
        if len(set(self.allowed_quality_flags)) != len(self.allowed_quality_flags):
            raise ValueError("detection.allowed_quality_flags must be unique")
        if any(not value.strip() for value in self.allowed_quality_flags):
            raise ValueError("detection.allowed_quality_flags must not contain blanks")


@dataclass(frozen=True)
class Phase5OutputConfig:
    directory: Path
    detections_filename: str
    survey_summary_filename: str

    def __post_init__(self) -> None:
        for name in ("detections_filename", "survey_summary_filename"):
            value = getattr(self, name)
            if not value.strip() or Path(value).name != value:
                raise ValueError(f"output.{name} must be a simple file name")


@dataclass(frozen=True)
class Phase5Config:
    source_path: Path
    phase4: Phase4Config
    rule: DetectionRule
    output: Phase5OutputConfig


@dataclass(frozen=True)
class DetectionRecord:
    """Classification derived only from one exported observation."""

    survey_id: str
    survey_date: date
    zone_id: str
    status: str
    anomaly_score: float | None
    uncertainty: float | None
    quality_flag: str
    reason: str

    def __post_init__(self) -> None:
        if self.status not in DETECTION_STATUSES:
            raise ValueError(f"status must be one of {DETECTION_STATUSES}")


@dataclass(frozen=True)
class DetectionSurveySummary:
    survey_id: str
    survey_date: date
    total_zones: int
    available_zones: int
    flagged_zones: int
    clear_zones: int
    unavailable_zones: int


@dataclass(frozen=True)
class Phase5Result:
    detections: tuple[DetectionRecord, ...]
    surveys: tuple[DetectionSurveySummary, ...]


def load_phase5_config(path: str | Path) -> Phase5Config:
    """Load strict Phase 5 detection settings."""

    source_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DetectionConfigError(f"unable to read {source_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise DetectionConfigError(f"invalid YAML in {source_path}: {exc}") from exc

    root = _mapping(raw, "configuration")
    _exact_keys(root, {"phase4_config", "detection", "output"}, "configuration")
    phase4 = load_phase4_config(
        _resolve_path(source_path.parent, root["phase4_config"])
    )
    detection = _mapping(root["detection"], "detection")
    _exact_keys(
        detection,
        {"anomaly_threshold", "maximum_uncertainty", "allowed_quality_flags"},
        "detection",
    )
    output = _mapping(root["output"], "output")
    _exact_keys(
        output,
        {"directory", "detections_filename", "survey_summary_filename"},
        "output",
    )

    try:
        rule = DetectionRule(
            anomaly_threshold=_number(
                detection["anomaly_threshold"], "detection.anomaly_threshold"
            ),
            maximum_uncertainty=_number(
                detection["maximum_uncertainty"], "detection.maximum_uncertainty"
            ),
            allowed_quality_flags=tuple(
                _text(value, "detection.allowed_quality_flags value")
                for value in _list(
                    detection["allowed_quality_flags"],
                    "detection.allowed_quality_flags",
                )
            ),
        )
        output_config = Phase5OutputConfig(
            directory=_resolve_path(source_path.parent, output["directory"]),
            detections_filename=_text(
                output["detections_filename"], "output.detections_filename"
            ),
            survey_summary_filename=_text(
                output["survey_summary_filename"],
                "output.survey_summary_filename",
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, DetectionConfigError):
            raise
        raise DetectionConfigError(str(exc)) from exc
    return Phase5Config(source_path, phase4, rule, output_config)


def classify_observations(
    observations: Sequence[Observation], rule: DetectionRule
) -> Phase5Result:
    """Classify observations without access to crop truth or issue inputs."""

    keys: set[tuple[str, str]] = set()
    records: list[DetectionRecord] = []
    for observation in observations:
        key = (observation.survey_id, observation.zone_id)
        if key in keys:
            raise ValueError(f"duplicate observation key: {key}")
        keys.add(key)
        status, reason = _classify(observation, rule)
        records.append(
            DetectionRecord(
                survey_id=observation.survey_id,
                survey_date=observation.survey_date,
                zone_id=observation.zone_id,
                status=status,
                anomaly_score=observation.anomaly_score,
                uncertainty=observation.uncertainty,
                quality_flag=observation.quality_flag,
                reason=reason,
            )
        )
    records.sort(key=lambda item: (item.survey_date, item.survey_id, item.zone_id))
    return Phase5Result(tuple(records), _summarize(records))


def _classify(observation: Observation, rule: DetectionRule) -> tuple[str, str]:
    if not observation.covered:
        return "unavailable", "no_coverage"
    if observation.anomaly_score is None or observation.uncertainty is None:
        return "unavailable", "missing_sensor_values"
    if observation.quality_flag not in rule.allowed_quality_flags:
        return "unavailable", f"quality_{observation.quality_flag}"
    if observation.uncertainty > rule.maximum_uncertainty:
        return "unavailable", "uncertainty_above_limit"
    if observation.anomaly_score >= rule.anomaly_threshold:
        return "flagged", "anomaly_threshold_met"
    return "clear", "anomaly_below_threshold"


def _summarize(
    records: Sequence[DetectionRecord],
) -> tuple[DetectionSurveySummary, ...]:
    grouped: dict[str, list[DetectionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.survey_id].append(record)
    summaries: list[DetectionSurveySummary] = []
    for survey_id, items in grouped.items():
        dates = {item.survey_date for item in items}
        if len(dates) != 1:
            raise ValueError(f"{survey_id} contains inconsistent survey dates")
        counts = {
            status: sum(item.status == status for item in items)
            for status in DETECTION_STATUSES
        }
        summaries.append(
            DetectionSurveySummary(
                survey_id=survey_id,
                survey_date=dates.pop(),
                total_zones=len(items),
                available_zones=counts["flagged"] + counts["clear"],
                flagged_zones=counts["flagged"],
                clear_zones=counts["clear"],
                unavailable_zones=counts["unavailable"],
            )
        )
    return tuple(sorted(summaries, key=lambda item: (item.survey_date, item.survey_id)))


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DetectionConfigError(f"{context} must be a mapping")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise DetectionConfigError(f"{context} must be a list")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = expected - set(mapping)
    extra = set(mapping) - expected
    if missing or extra:
        raise DetectionConfigError(
            f"{context} keys do not match schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _resolve_path(base: Path, value: Any) -> Path:
    path = Path(_text(value, "configured path"))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DetectionConfigError(f"{name} must be a nonempty string")
    return value.strip()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DetectionConfigError(f"{name} must be numeric")
    return float(value)


def _unit_fraction(value: float, name: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
