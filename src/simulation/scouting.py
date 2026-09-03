"""Conventional W-route scout observations for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite
from pathlib import Path
from random import Random
from typing import Any, Mapping

import yaml

from src.farm import Farm, load_farm
from src.simulation.observations import Phase4Config, load_phase4_config


class ScoutConfigError(ValueError):
    """Raised when Phase 6 configuration is invalid."""


@dataclass(frozen=True)
class ScoutSettings:
    route_zone_ids: tuple[str, ...]
    seed: int
    healthy_visual_baseline: float
    visibility_sensitivity: float
    visual_noise_std: float
    missing_probability: float
    inspection_minutes_per_zone: float
    travel_minutes_between_zones: float
    missed_days: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.route_zone_ids:
            raise ValueError("scout.route_zone_ids must not be empty")
        if len(set(self.route_zone_ids)) != len(self.route_zone_ids):
            raise ValueError("scout.route_zone_ids must be unique")
        if any(not zone_id.strip() for zone_id in self.route_zone_ids):
            raise ValueError("scout.route_zone_ids must not contain blanks")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("scout.seed must be an integer")
        for name in (
            "healthy_visual_baseline",
            "visibility_sensitivity",
            "missing_probability",
        ):
            _unit_fraction(getattr(self, name), f"scout.{name}")
        _nonnegative_number(self.visual_noise_std, "scout.visual_noise_std")
        _positive_number(
            self.inspection_minutes_per_zone,
            "scout.inspection_minutes_per_zone",
        )
        _nonnegative_number(
            self.travel_minutes_between_zones,
            "scout.travel_minutes_between_zones",
        )
        if len(set(self.missed_days)) != len(self.missed_days):
            raise ValueError("scout.missed_days must be unique")
        for day in self.missed_days:
            _nonnegative_integer(day, "scout.missed_days value")


@dataclass(frozen=True)
class Phase6OutputConfig:
    directory: Path
    observations_filename: str
    survey_summary_filename: str

    def __post_init__(self) -> None:
        for name in ("observations_filename", "survey_summary_filename"):
            value = getattr(self, name)
            if not value.strip() or Path(value).name != value:
                raise ValueError(f"output.{name} must be a simple file name")


@dataclass(frozen=True)
class Phase6Config:
    source_path: Path
    phase4: Phase4Config
    scout: ScoutSettings
    output: Phase6OutputConfig


@dataclass(frozen=True)
class ScoutObservation:
    """One visual observation with no hidden issue fields."""

    survey_id: str
    survey_date: date
    zone_id: str
    visit_order: int
    visit_elapsed_minutes: float | None
    visual_anomaly_score: float | None
    uncertainty: float | None
    quality_flag: str


@dataclass(frozen=True)
class ScoutSurveySummary:
    survey_id: str
    survey_date: date
    campaign_day: int
    missed_survey: bool
    planned_zones: int
    visited_zones: int
    valid_observations: int
    unavailable_observations: int
    route_duration_minutes: float


@dataclass(frozen=True)
class Phase6Result:
    observations: tuple[ScoutObservation, ...]
    surveys: tuple[ScoutSurveySummary, ...]


def load_phase6_config(path: str | Path) -> Phase6Config:
    """Load strict Phase 6 scout settings."""

    source_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ScoutConfigError(f"unable to read {source_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ScoutConfigError(f"invalid YAML in {source_path}: {exc}") from exc

    root = _mapping(raw, "configuration")
    _exact_keys(root, {"phase4_config", "scout", "output"}, "configuration")
    phase4 = load_phase4_config(
        _resolve_path(source_path.parent, root["phase4_config"])
    )
    scout_raw = _mapping(root["scout"], "scout")
    scout_keys = {
        "route_zone_ids",
        "seed",
        "healthy_visual_baseline",
        "visibility_sensitivity",
        "visual_noise_std",
        "missing_probability",
        "inspection_minutes_per_zone",
        "travel_minutes_between_zones",
        "missed_days",
    }
    _exact_keys(scout_raw, scout_keys, "scout")
    output_raw = _mapping(root["output"], "output")
    _exact_keys(
        output_raw,
        {"directory", "observations_filename", "survey_summary_filename"},
        "output",
    )
    try:
        scout = ScoutSettings(
            route_zone_ids=tuple(
                _text(value, "scout.route_zone_ids value")
                for value in _list(
                    scout_raw["route_zone_ids"], "scout.route_zone_ids"
                )
            ),
            seed=_integer(scout_raw["seed"], "scout.seed"),
            healthy_visual_baseline=_number(
                scout_raw["healthy_visual_baseline"],
                "scout.healthy_visual_baseline",
            ),
            visibility_sensitivity=_number(
                scout_raw["visibility_sensitivity"],
                "scout.visibility_sensitivity",
            ),
            visual_noise_std=_number(
                scout_raw["visual_noise_std"], "scout.visual_noise_std"
            ),
            missing_probability=_number(
                scout_raw["missing_probability"], "scout.missing_probability"
            ),
            inspection_minutes_per_zone=_number(
                scout_raw["inspection_minutes_per_zone"],
                "scout.inspection_minutes_per_zone",
            ),
            travel_minutes_between_zones=_number(
                scout_raw["travel_minutes_between_zones"],
                "scout.travel_minutes_between_zones",
            ),
            missed_days=tuple(
                _integer(value, "scout.missed_days value")
                for value in _list(scout_raw["missed_days"], "scout.missed_days")
            ),
        )
        output = Phase6OutputConfig(
            directory=_resolve_path(source_path.parent, output_raw["directory"]),
            observations_filename=_text(
                output_raw["observations_filename"],
                "output.observations_filename",
            ),
            survey_summary_filename=_text(
                output_raw["survey_summary_filename"],
                "output.survey_summary_filename",
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ScoutConfigError):
            raise
        raise ScoutConfigError(str(exc)) from exc

    farm = load_farm(phase4.phase1)
    _validate_scout(scout, farm=farm, scheduled_days=phase4.schedule.survey_days)
    return Phase6Config(source_path, phase4, scout, output)


def simulate_scouting(
    config: Phase6Config, *, farm: Farm | None = None
) -> Phase6Result:
    """Generate reproducible visual observations along the configured route."""

    farm = farm or load_farm(config.phase4.phase1)
    _validate_scout(
        config.scout,
        farm=farm,
        scheduled_days=config.phase4.schedule.survey_days,
    )
    issues_by_zone = {
        zone_id: issue
        for issue in config.phase4.issues
        for zone_id in issue.zone_ids
    }
    random = Random(config.scout.seed)
    observations: list[ScoutObservation] = []
    summaries: list[ScoutSurveySummary] = []
    sowing_date = config.phase4.phase1.calendar.base_sowing_date
    duration = _route_duration(config.scout)

    for survey_number, campaign_day in enumerate(
        config.phase4.schedule.survey_days, start=1
    ):
        survey_id = f"SURVEY_{survey_number:03d}"
        survey_date = sowing_date + timedelta(days=campaign_day)
        missed_survey = campaign_day in config.scout.missed_days
        survey_records: list[ScoutObservation] = []
        for visit_order, zone_id in enumerate(config.scout.route_zone_ids, start=1):
            elapsed = (
                visit_order * config.scout.inspection_minutes_per_zone
                + (visit_order - 1) * config.scout.travel_minutes_between_zones
            )
            if missed_survey:
                record = ScoutObservation(
                    survey_id,
                    survey_date,
                    zone_id,
                    visit_order,
                    None,
                    None,
                    None,
                    "missed_survey",
                )
            elif random.random() < config.scout.missing_probability:
                record = ScoutObservation(
                    survey_id,
                    survey_date,
                    zone_id,
                    visit_order,
                    elapsed,
                    None,
                    1.0,
                    "not_observed",
                )
            else:
                issue = issues_by_zone.get(zone_id)
                visibility = issue.visibility_on(campaign_day) if issue else 0.0
                score = _clip(
                    config.scout.healthy_visual_baseline
                    + config.scout.visibility_sensitivity * visibility
                    + random.gauss(0.0, config.scout.visual_noise_std)
                )
                uncertainty = _clip(config.scout.visual_noise_std * 3.0)
                record = ScoutObservation(
                    survey_id,
                    survey_date,
                    zone_id,
                    visit_order,
                    elapsed,
                    score,
                    uncertainty,
                    "good",
                )
            survey_records.append(record)
            observations.append(record)

        valid = sum(item.visual_anomaly_score is not None for item in survey_records)
        summaries.append(
            ScoutSurveySummary(
                survey_id=survey_id,
                survey_date=survey_date,
                campaign_day=campaign_day,
                missed_survey=missed_survey,
                planned_zones=len(config.scout.route_zone_ids),
                visited_zones=0 if missed_survey else len(config.scout.route_zone_ids),
                valid_observations=valid,
                unavailable_observations=len(survey_records) - valid,
                route_duration_minutes=0.0 if missed_survey else duration,
            )
        )
    return Phase6Result(tuple(observations), tuple(summaries))


def _route_duration(settings: ScoutSettings) -> float:
    return (
        len(settings.route_zone_ids) * settings.inspection_minutes_per_zone
        + max(0, len(settings.route_zone_ids) - 1)
        * settings.travel_minutes_between_zones
    )


def _validate_scout(
    settings: ScoutSettings, *, farm: Farm, scheduled_days: tuple[int, ...]
) -> None:
    known_zones = {zone.zone_id for zone in farm.zones}
    unknown = set(settings.route_zone_ids) - known_zones
    if unknown:
        raise ScoutConfigError(f"scout route contains unknown zones: {sorted(unknown)}")
    unscheduled = set(settings.missed_days) - set(scheduled_days)
    if unscheduled:
        raise ScoutConfigError(
            f"scout.missed_days are not scheduled survey days: {sorted(unscheduled)}"
        )


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScoutConfigError(f"{context} must be a mapping")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ScoutConfigError(f"{context} must be a list")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = expected - set(mapping)
    extra = set(mapping) - expected
    if missing or extra:
        raise ScoutConfigError(
            f"{context} keys do not match schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _resolve_path(base: Path, value: Any) -> Path:
    path = Path(_text(value, "configured path"))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScoutConfigError(f"{name} must be a nonempty string")
    return value.strip()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoutConfigError(f"{name} must be numeric")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ScoutConfigError(f"{name} must be an integer")
    return value


def _positive_number(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _nonnegative_number(value: float, name: str) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _nonnegative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _unit_fraction(value: float, name: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
