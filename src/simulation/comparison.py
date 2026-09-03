"""Truth-separated drone and scout detection comparison for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.simulation.detection import (
    DetectionRule,
    Phase5Config,
    Phase5Result,
    load_phase5_config,
)
from src.simulation.issues import IssueScenario
from src.simulation.scouting import (
    Phase6Config,
    Phase6Result,
    ScoutObservation,
    load_phase6_config,
)


METHODS = ("drone", "scout")
EVALUATION_LABELS = (
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "unavailable",
)


class ComparisonConfigError(ValueError):
    """Raised when Phase 7 configuration is invalid."""


@dataclass(frozen=True)
class Phase7OutputConfig:
    directory: Path
    method_detections_filename: str
    detection_evaluation_filename: str
    issue_summary_filename: str

    def __post_init__(self) -> None:
        for name in (
            "method_detections_filename",
            "detection_evaluation_filename",
            "issue_summary_filename",
        ):
            value = getattr(self, name)
            if not value.strip() or Path(value).name != value:
                raise ValueError(f"output.{name} must be a simple file name")


@dataclass(frozen=True)
class Phase7Config:
    source_path: Path
    phase5: Phase5Config
    phase6: Phase6Config
    scout_rule: DetectionRule
    output: Phase7OutputConfig


@dataclass(frozen=True)
class MethodDetection:
    """One truth-blind method result."""

    method: str
    survey_id: str
    survey_date: date
    zone_id: str
    status: str
    score: float | None
    uncertainty: float | None
    quality_flag: str
    reason: str

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.status not in {"flagged", "clear", "unavailable"}:
            raise ValueError("invalid detection status")


@dataclass(frozen=True)
class DetectionEvaluation:
    method: str
    survey_id: str
    survey_date: date
    zone_id: str
    status: str
    truth_positive: bool
    evaluation: str

    def __post_init__(self) -> None:
        if self.evaluation not in EVALUATION_LABELS:
            raise ValueError(f"evaluation must be one of {EVALUATION_LABELS}")


@dataclass(frozen=True)
class IssueDetectionSummary:
    issue_id: str
    mechanism: str
    method: str
    onset_date: date
    footprint_zone_count: int
    observable_zone_count: int
    detected: bool
    first_detection_date: date | None
    first_detection_zone: str
    detection_delay_days: int | None


@dataclass(frozen=True)
class Phase7Result:
    method_detections: tuple[MethodDetection, ...]
    evaluations: tuple[DetectionEvaluation, ...]
    issue_summaries: tuple[IssueDetectionSummary, ...]


def load_phase7_config(path: str | Path) -> Phase7Config:
    """Load strict Phase 7 comparison settings."""

    source_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ComparisonConfigError(f"unable to read {source_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ComparisonConfigError(f"invalid YAML in {source_path}: {exc}") from exc

    root = _mapping(raw, "configuration")
    _exact_keys(
        root,
        {"phase5_config", "phase6_config", "scout_detection", "output"},
        "configuration",
    )
    phase5 = load_phase5_config(
        _resolve_path(source_path.parent, root["phase5_config"])
    )
    phase6 = load_phase6_config(
        _resolve_path(source_path.parent, root["phase6_config"])
    )
    if phase5.phase4.source_path != phase6.phase4.source_path:
        raise ComparisonConfigError("Phase 5 and Phase 6 must use the same Phase 4")

    scout = _mapping(root["scout_detection"], "scout_detection")
    _exact_keys(
        scout,
        {"anomaly_threshold", "maximum_uncertainty", "allowed_quality_flags"},
        "scout_detection",
    )
    output = _mapping(root["output"], "output")
    _exact_keys(
        output,
        {
            "directory",
            "method_detections_filename",
            "detection_evaluation_filename",
            "issue_summary_filename",
        },
        "output",
    )
    try:
        scout_rule = DetectionRule(
            anomaly_threshold=_number(
                scout["anomaly_threshold"],
                "scout_detection.anomaly_threshold",
            ),
            maximum_uncertainty=_number(
                scout["maximum_uncertainty"],
                "scout_detection.maximum_uncertainty",
            ),
            allowed_quality_flags=tuple(
                _text(value, "scout_detection.allowed_quality_flags value")
                for value in _list(
                    scout["allowed_quality_flags"],
                    "scout_detection.allowed_quality_flags",
                )
            ),
        )
        output_config = Phase7OutputConfig(
            directory=_resolve_path(source_path.parent, output["directory"]),
            method_detections_filename=_text(
                output["method_detections_filename"],
                "output.method_detections_filename",
            ),
            detection_evaluation_filename=_text(
                output["detection_evaluation_filename"],
                "output.detection_evaluation_filename",
            ),
            issue_summary_filename=_text(
                output["issue_summary_filename"],
                "output.issue_summary_filename",
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ComparisonConfigError):
            raise
        raise ComparisonConfigError(str(exc)) from exc
    return Phase7Config(source_path, phase5, phase6, scout_rule, output_config)


def compare_methods(
    config: Phase7Config,
    drone_result: Phase5Result,
    scout_result: Phase6Result,
) -> Phase7Result:
    """Combine truth-blind detections, then evaluate them against issue truth."""

    detections = _drone_detections(drone_result) + _scout_detections(
        scout_result, config.scout_rule
    )
    detections = tuple(
        sorted(
            detections,
            key=lambda item: (
                item.method,
                item.survey_date,
                item.survey_id,
                item.zone_id,
            ),
        )
    )
    _validate_method_keys(detections)
    issues = config.phase5.phase4.issues
    sowing_date = config.phase5.phase4.phase1.calendar.base_sowing_date
    evaluations = _evaluate(detections, issues=issues, sowing_date=sowing_date)
    summaries = _issue_summaries(
        detections,
        issues=issues,
        sowing_date=sowing_date,
    )
    return Phase7Result(detections, evaluations, summaries)


def _drone_detections(result: Phase5Result) -> tuple[MethodDetection, ...]:
    return tuple(
        MethodDetection(
            method="drone",
            survey_id=item.survey_id,
            survey_date=item.survey_date,
            zone_id=item.zone_id,
            status=item.status,
            score=item.anomaly_score,
            uncertainty=item.uncertainty,
            quality_flag=item.quality_flag,
            reason=item.reason,
        )
        for item in result.detections
    )


def _scout_detections(
    result: Phase6Result, rule: DetectionRule
) -> tuple[MethodDetection, ...]:
    records: list[MethodDetection] = []
    for item in result.observations:
        status, reason = _classify_scout(item, rule)
        records.append(
            MethodDetection(
                method="scout",
                survey_id=item.survey_id,
                survey_date=item.survey_date,
                zone_id=item.zone_id,
                status=status,
                score=item.visual_anomaly_score,
                uncertainty=item.uncertainty,
                quality_flag=item.quality_flag,
                reason=reason,
            )
        )
    return tuple(records)


def _classify_scout(
    observation: ScoutObservation, rule: DetectionRule
) -> tuple[str, str]:
    if observation.visual_anomaly_score is None or observation.uncertainty is None:
        return "unavailable", "missing_visual_observation"
    if observation.quality_flag not in rule.allowed_quality_flags:
        return "unavailable", f"quality_{observation.quality_flag}"
    if observation.uncertainty > rule.maximum_uncertainty:
        return "unavailable", "uncertainty_above_limit"
    if observation.visual_anomaly_score >= rule.anomaly_threshold:
        return "flagged", "visual_threshold_met"
    return "clear", "visual_below_threshold"


def _evaluate(
    detections: Sequence[MethodDetection],
    *,
    issues: Sequence[IssueScenario],
    sowing_date: date,
) -> tuple[DetectionEvaluation, ...]:
    issue_by_zone = {
        zone_id: issue for issue in issues for zone_id in issue.zone_ids
    }
    records: list[DetectionEvaluation] = []
    for item in detections:
        issue = issue_by_zone.get(item.zone_id)
        truth_positive = (
            issue is not None
            and item.survey_date >= sowing_date + timedelta(days=issue.onset_day)
        )
        if item.status == "unavailable":
            evaluation = "unavailable"
        elif truth_positive and item.status == "flagged":
            evaluation = "true_positive"
        elif truth_positive:
            evaluation = "false_negative"
        elif item.status == "flagged":
            evaluation = "false_positive"
        else:
            evaluation = "true_negative"
        records.append(
            DetectionEvaluation(
                method=item.method,
                survey_id=item.survey_id,
                survey_date=item.survey_date,
                zone_id=item.zone_id,
                status=item.status,
                truth_positive=truth_positive,
                evaluation=evaluation,
            )
        )
    return tuple(records)


def _issue_summaries(
    detections: Sequence[MethodDetection],
    *,
    issues: Sequence[IssueScenario],
    sowing_date: date,
) -> tuple[IssueDetectionSummary, ...]:
    summaries: list[IssueDetectionSummary] = []
    for issue in issues:
        onset_date = sowing_date + timedelta(days=issue.onset_day)
        footprint = set(issue.zone_ids)
        for method in METHODS:
            method_rows = [
                item
                for item in detections
                if item.method == method and item.zone_id in footprint
            ]
            observable_zones = {item.zone_id for item in method_rows}
            flags = [
                item
                for item in method_rows
                if item.status == "flagged" and item.survey_date >= onset_date
            ]
            first = min(
                flags,
                key=lambda item: (item.survey_date, item.survey_id, item.zone_id),
                default=None,
            )
            summaries.append(
                IssueDetectionSummary(
                    issue_id=issue.issue_id,
                    mechanism=issue.mechanism,
                    method=method,
                    onset_date=onset_date,
                    footprint_zone_count=len(footprint),
                    observable_zone_count=len(observable_zones),
                    detected=first is not None,
                    first_detection_date=first.survey_date if first else None,
                    first_detection_zone=first.zone_id if first else "",
                    detection_delay_days=(first.survey_date - onset_date).days
                    if first
                    else None,
                )
            )
    return tuple(
        sorted(summaries, key=lambda item: (item.issue_id, item.method))
    )


def _validate_method_keys(detections: Sequence[MethodDetection]) -> None:
    keys: set[tuple[str, str, str]] = set()
    for item in detections:
        key = (item.method, item.survey_id, item.zone_id)
        if key in keys:
            raise ValueError(f"duplicate method detection key: {key}")
        keys.add(key)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonConfigError(f"{context} must be a mapping")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ComparisonConfigError(f"{context} must be a list")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = expected - set(mapping)
    extra = set(mapping) - expected
    if missing or extra:
        raise ComparisonConfigError(
            f"{context} keys do not match schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _resolve_path(base: Path, value: Any) -> Path:
    path = Path(_text(value, "configured path"))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ComparisonConfigError(f"{name} must be a nonempty string")
    return value.strip()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonConfigError(f"{name} must be numeric")
    return float(value)
