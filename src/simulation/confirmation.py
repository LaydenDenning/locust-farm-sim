"""Seeded human confirmation of Phase 7 anomaly flags."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

import yaml

from src.simulation.comparison import (
    METHODS,
    Phase7Config,
    Phase7Result,
    load_phase7_config,
)
from src.simulation.issues import IssueScenario


CONFIRMATION_EVALUATIONS = (
    "true_confirmation",
    "false_confirmation",
    "correct_rejection",
    "missed_confirmation",
)


class ConfirmationConfigError(ValueError):
    """Raised when Phase 8 configuration is invalid."""


@dataclass(frozen=True)
class ConfirmationSettings:
    seed: int
    drone_delay_days: int
    scout_delay_days: int
    sensitivity: float
    specificity: float

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("confirmation.seed must be an integer")
        _nonnegative_integer(
            self.drone_delay_days, "confirmation.drone_delay_days"
        )
        _nonnegative_integer(
            self.scout_delay_days, "confirmation.scout_delay_days"
        )
        _unit_fraction(self.sensitivity, "confirmation.sensitivity")
        _unit_fraction(self.specificity, "confirmation.specificity")

    def delay_for(self, method: str) -> int:
        if method == "drone":
            return self.drone_delay_days
        if method == "scout":
            return self.scout_delay_days
        raise ValueError(f"unknown method: {method}")


@dataclass(frozen=True)
class Phase8OutputConfig:
    directory: Path
    confirmation_events_filename: str
    confirmation_evaluation_filename: str
    issue_summary_filename: str

    def __post_init__(self) -> None:
        for name in (
            "confirmation_events_filename",
            "confirmation_evaluation_filename",
            "issue_summary_filename",
        ):
            value = getattr(self, name)
            if not value.strip() or Path(value).name != value:
                raise ValueError(f"output.{name} must be a simple file name")


@dataclass(frozen=True)
class Phase8Config:
    source_path: Path
    phase7: Phase7Config
    confirmation: ConfirmationSettings
    output: Phase8OutputConfig


@dataclass(frozen=True)
class ConfirmationEvent:
    """Operational result without hidden truth labels."""

    confirmation_id: str
    method: str
    survey_id: str
    zone_id: str
    flag_date: date
    confirmation_date: date
    result: str

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.result not in {"confirmed", "rejected"}:
            raise ValueError("confirmation result must be confirmed or rejected")
        if self.confirmation_date < self.flag_date:
            raise ValueError("confirmation_date must not precede flag_date")


@dataclass(frozen=True)
class ConfirmationEvaluation:
    confirmation_id: str
    method: str
    survey_id: str
    zone_id: str
    flag_date: date
    confirmation_date: date
    truth_positive: bool
    evaluation: str

    def __post_init__(self) -> None:
        if self.evaluation not in CONFIRMATION_EVALUATIONS:
            raise ValueError(
                f"evaluation must be one of {CONFIRMATION_EVALUATIONS}"
            )


@dataclass(frozen=True)
class IssueConfirmationSummary:
    issue_id: str
    mechanism: str
    method: str
    onset_date: date
    confirmation_request_count: int
    confirmed: bool
    first_confirmation_date: date | None
    first_confirmation_zone: str
    confirmation_delay_days: int | None


@dataclass(frozen=True)
class Phase8Result:
    confirmation_events: tuple[ConfirmationEvent, ...]
    evaluations: tuple[ConfirmationEvaluation, ...]
    issue_summaries: tuple[IssueConfirmationSummary, ...]


def load_phase8_config(path: str | Path) -> Phase8Config:
    """Load strict Phase 8 confirmation settings."""

    source_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfirmationConfigError(f"unable to read {source_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfirmationConfigError(f"invalid YAML in {source_path}: {exc}") from exc

    root = _mapping(raw, "configuration")
    _exact_keys(root, {"phase7_config", "confirmation", "output"}, "configuration")
    phase7 = load_phase7_config(
        _resolve_path(source_path.parent, root["phase7_config"])
    )
    confirmation = _mapping(root["confirmation"], "confirmation")
    _exact_keys(
        confirmation,
        {"seed", "drone_delay_days", "scout_delay_days", "sensitivity", "specificity"},
        "confirmation",
    )
    output = _mapping(root["output"], "output")
    _exact_keys(
        output,
        {
            "directory",
            "confirmation_events_filename",
            "confirmation_evaluation_filename",
            "issue_summary_filename",
        },
        "output",
    )
    try:
        settings = ConfirmationSettings(
            seed=_integer(confirmation["seed"], "confirmation.seed"),
            drone_delay_days=_integer(
                confirmation["drone_delay_days"],
                "confirmation.drone_delay_days",
            ),
            scout_delay_days=_integer(
                confirmation["scout_delay_days"],
                "confirmation.scout_delay_days",
            ),
            sensitivity=_number(
                confirmation["sensitivity"], "confirmation.sensitivity"
            ),
            specificity=_number(
                confirmation["specificity"], "confirmation.specificity"
            ),
        )
        output_config = Phase8OutputConfig(
            directory=_resolve_path(source_path.parent, output["directory"]),
            confirmation_events_filename=_text(
                output["confirmation_events_filename"],
                "output.confirmation_events_filename",
            ),
            confirmation_evaluation_filename=_text(
                output["confirmation_evaluation_filename"],
                "output.confirmation_evaluation_filename",
            ),
            issue_summary_filename=_text(
                output["issue_summary_filename"],
                "output.issue_summary_filename",
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfirmationConfigError):
            raise
        raise ConfirmationConfigError(str(exc)) from exc
    return Phase8Config(source_path, phase7, settings, output_config)


def simulate_confirmations(
    config: Phase8Config, phase7_result: Phase7Result
) -> Phase8Result:
    """Confirm every flag, then evaluate the human result separately."""

    evaluations_by_key = {
        (item.method, item.survey_id, item.zone_id): item
        for item in phase7_result.evaluations
    }
    if len(evaluations_by_key) != len(phase7_result.evaluations):
        raise ValueError("Phase 7 contains duplicate evaluation keys")

    flagged = sorted(
        (
            item
            for item in phase7_result.method_detections
            if item.status == "flagged"
        ),
        key=lambda item: (item.flag_date if hasattr(item, "flag_date") else item.survey_date, item.method, item.survey_id, item.zone_id),
    )
    random = Random(config.confirmation.seed)
    events: list[ConfirmationEvent] = []
    evaluations: list[ConfirmationEvaluation] = []
    for index, item in enumerate(flagged, start=1):
        key = (item.method, item.survey_id, item.zone_id)
        if key not in evaluations_by_key:
            raise ValueError(f"flag has no Phase 7 evaluation: {key}")
        truth_positive = evaluations_by_key[key].truth_positive
        confirmation_probability = (
            config.confirmation.sensitivity
            if truth_positive
            else 1.0 - config.confirmation.specificity
        )
        confirmed = random.random() < confirmation_probability
        confirmation_id = f"CONF_{index:04d}"
        confirmation_date = item.survey_date + timedelta(
            days=config.confirmation.delay_for(item.method)
        )
        result = "confirmed" if confirmed else "rejected"
        events.append(
            ConfirmationEvent(
                confirmation_id,
                item.method,
                item.survey_id,
                item.zone_id,
                item.survey_date,
                confirmation_date,
                result,
            )
        )
        evaluations.append(
            ConfirmationEvaluation(
                confirmation_id,
                item.method,
                item.survey_id,
                item.zone_id,
                item.survey_date,
                confirmation_date,
                truth_positive,
                _evaluation_label(confirmed, truth_positive),
            )
        )

    summaries = _issue_summaries(
        events,
        evaluations,
        issues=config.phase7.phase5.phase4.issues,
        sowing_date=config.phase7.phase5.phase4.phase1.calendar.base_sowing_date,
    )
    return Phase8Result(tuple(events), tuple(evaluations), summaries)


def _evaluation_label(confirmed: bool, truth_positive: bool) -> str:
    if confirmed and truth_positive:
        return "true_confirmation"
    if confirmed:
        return "false_confirmation"
    if truth_positive:
        return "missed_confirmation"
    return "correct_rejection"


def _issue_summaries(
    events: Sequence[ConfirmationEvent],
    evaluations: Sequence[ConfirmationEvaluation],
    *,
    issues: Sequence[IssueScenario],
    sowing_date: date,
) -> tuple[IssueConfirmationSummary, ...]:
    truth_by_id = {item.confirmation_id: item.truth_positive for item in evaluations}
    summaries: list[IssueConfirmationSummary] = []
    for issue in issues:
        footprint = set(issue.zone_ids)
        onset_date = sowing_date + timedelta(days=issue.onset_day)
        for method in METHODS:
            requests = [
                item
                for item in events
                if item.method == method
                and item.zone_id in footprint
                and item.flag_date >= onset_date
            ]
            true_confirmations = [
                item
                for item in requests
                if item.result == "confirmed" and truth_by_id[item.confirmation_id]
            ]
            first = min(
                true_confirmations,
                key=lambda item: (item.confirmation_date, item.zone_id),
                default=None,
            )
            summaries.append(
                IssueConfirmationSummary(
                    issue_id=issue.issue_id,
                    mechanism=issue.mechanism,
                    method=method,
                    onset_date=onset_date,
                    confirmation_request_count=len(requests),
                    confirmed=first is not None,
                    first_confirmation_date=first.confirmation_date if first else None,
                    first_confirmation_zone=first.zone_id if first else "",
                    confirmation_delay_days=(first.confirmation_date - onset_date).days
                    if first
                    else None,
                )
            )
    return tuple(
        sorted(summaries, key=lambda item: (item.issue_id, item.method))
    )


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfirmationConfigError(f"{context} must be a mapping")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = expected - set(mapping)
    extra = set(mapping) - expected
    if missing or extra:
        raise ConfirmationConfigError(
            f"{context} keys do not match schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _resolve_path(base: Path, value: Any) -> Path:
    path = Path(_text(value, "configured path"))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfirmationConfigError(f"{name} must be a nonempty string")
    return value.strip()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfirmationConfigError(f"{name} must be numeric")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfirmationConfigError(f"{name} must be an integer")
    return value


def _nonnegative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _unit_fraction(value: float, name: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
