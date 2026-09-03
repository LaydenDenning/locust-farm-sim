"""Seeded, zone-level RGB/NIR observations for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from math import exp, isfinite
from pathlib import Path
from random import Random
from statistics import median
from typing import Any, Mapping

import pandas as pd
import yaml

from src.farm import Farm, Phase1Config, load_farm, load_phase1_config
from src.simulation.drone import MissionPlan, Phase3Config, load_phase3_config, plan_mission
from src.simulation.issues import IssueScenario, load_issue_scenarios


class ObservationConfigError(ValueError):
    """Raised when Phase 4 configuration is invalid."""


@dataclass(frozen=True)
class SurveyScheduleConfig:
    start_day: int
    end_day: int
    interval_days: int
    grounded_days: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("start_day", "end_day"):
            _nonnegative_integer(getattr(self, name), f"schedule.{name}")
        _positive_integer(self.interval_days, "schedule.interval_days")
        if self.end_day < self.start_day:
            raise ValueError("schedule.end_day must not precede start_day")
        if len(set(self.grounded_days)) != len(self.grounded_days):
            raise ValueError("schedule.grounded_days must be unique")
        for day in self.grounded_days:
            _nonnegative_integer(day, "schedule.grounded_days value")
            if day < self.start_day or day > self.end_day:
                raise ValueError("grounded days must fall within the survey schedule")

    @property
    def survey_days(self) -> tuple[int, ...]:
        return tuple(range(self.start_day, self.end_day + 1, self.interval_days))


@dataclass(frozen=True)
class SensorNoiseConfig:
    seed: int
    random_noise_std: float
    illumination_std: float
    rgb_bias: float
    nir_bias: float
    spatial_mix_fraction: float
    registration_mix_fraction: float
    cloud_probability: float
    missing_probability: float
    cloud_signal_factor: float
    lai_saturation: float

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("observation.seed must be an integer")
        for name in (
            "random_noise_std",
            "illumination_std",
            "rgb_bias",
            "nir_bias",
        ):
            _nonnegative_number(getattr(self, name), f"observation.{name}")
        for name in (
            "spatial_mix_fraction",
            "registration_mix_fraction",
            "cloud_probability",
            "missing_probability",
        ):
            _unit_fraction(getattr(self, name), f"observation.{name}")
        if not isfinite(self.cloud_signal_factor) or not 0 < self.cloud_signal_factor <= 1:
            raise ValueError("observation.cloud_signal_factor must be above 0 and at most 1")
        _positive_number(self.lai_saturation, "observation.lai_saturation")


@dataclass(frozen=True)
class Phase4OutputConfig:
    directory: Path
    observations_filename: str
    survey_summary_filename: str

    def __post_init__(self) -> None:
        for name in ("observations_filename", "survey_summary_filename"):
            value = getattr(self, name)
            if not value.strip() or Path(value).name != value:
                raise ValueError(f"output.{name} must be a simple file name")


@dataclass(frozen=True)
class Phase4Config:
    source_path: Path
    phase1: Phase1Config
    phase3: Phase3Config
    issues_file: Path
    issues: tuple[IssueScenario, ...]
    schedule: SurveyScheduleConfig
    noise: SensorNoiseConfig
    output: Phase4OutputConfig


@dataclass(frozen=True)
class Observation:
    """One sensor-only record; hidden issue fields are deliberately absent."""

    survey_id: str
    survey_date: date
    zone_id: str
    covered: bool
    coverage_fraction: float
    crop_active: bool
    relative_red: float | None
    relative_green: float | None
    relative_blue: float | None
    relative_nir: float | None
    ndvi_like: float | None
    canopy_cover: float | None
    anomaly_score: float | None
    uncertainty: float | None
    quality_flag: str


@dataclass(frozen=True)
class SurveySummary:
    survey_id: str
    survey_date: date
    campaign_day: int
    grounded: bool
    sortie_count: int
    covered_zones: int
    valid_observations: int
    cloudy_observations: int
    missing_observations: int
    total_image_count: int


@dataclass(frozen=True)
class Phase4Result:
    observations: tuple[Observation, ...]
    surveys: tuple[SurveySummary, ...]


def load_phase4_config(path: str | Path) -> Phase4Config:
    """Load Phase 4 settings and their Phase 1/3 inputs."""

    source_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ObservationConfigError(f"unable to read {source_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ObservationConfigError(f"invalid YAML in {source_path}: {exc}") from exc
    root = _mapping(raw, "configuration")
    _exact_keys(
        root,
        {"phase1_config", "phase3_config", "issues_file", "schedule", "observation", "output"},
        "configuration",
    )
    phase1 = load_phase1_config(_resolve_path(source_path.parent, root["phase1_config"]))
    phase3 = load_phase3_config(_resolve_path(source_path.parent, root["phase3_config"]))
    if phase1.source_path != phase3.phase1.source_path:
        raise ObservationConfigError("Phase 1 and Phase 3 must reference the same farm")
    farm = load_farm(phase1)
    issues_file = _resolve_path(source_path.parent, root["issues_file"])

    schedule_raw = _mapping(root["schedule"], "schedule")
    _exact_keys(schedule_raw, {"start_day", "end_day", "interval_days", "grounded_days"}, "schedule")
    noise_raw = _mapping(root["observation"], "observation")
    noise_keys = {
        "seed",
        "random_noise_std",
        "illumination_std",
        "rgb_bias",
        "nir_bias",
        "spatial_mix_fraction",
        "registration_mix_fraction",
        "cloud_probability",
        "missing_probability",
        "cloud_signal_factor",
        "lai_saturation",
    }
    _exact_keys(noise_raw, noise_keys, "observation")
    output_raw = _mapping(root["output"], "output")
    _exact_keys(output_raw, {"directory", "observations_filename", "survey_summary_filename"}, "output")

    try:
        schedule = SurveyScheduleConfig(
            start_day=_integer(schedule_raw["start_day"], "schedule.start_day"),
            end_day=_integer(schedule_raw["end_day"], "schedule.end_day"),
            interval_days=_integer(schedule_raw["interval_days"], "schedule.interval_days"),
            grounded_days=tuple(
                _integer(day, "schedule.grounded_days value")
                for day in _list(schedule_raw["grounded_days"], "schedule.grounded_days")
            ),
        )
        noise = SensorNoiseConfig(
            seed=_integer(noise_raw["seed"], "observation.seed"),
            random_noise_std=_number(noise_raw["random_noise_std"], "observation.random_noise_std"),
            illumination_std=_number(noise_raw["illumination_std"], "observation.illumination_std"),
            rgb_bias=_number(noise_raw["rgb_bias"], "observation.rgb_bias"),
            nir_bias=_number(noise_raw["nir_bias"], "observation.nir_bias"),
            spatial_mix_fraction=_number(noise_raw["spatial_mix_fraction"], "observation.spatial_mix_fraction"),
            registration_mix_fraction=_number(noise_raw["registration_mix_fraction"], "observation.registration_mix_fraction"),
            cloud_probability=_number(noise_raw["cloud_probability"], "observation.cloud_probability"),
            missing_probability=_number(noise_raw["missing_probability"], "observation.missing_probability"),
            cloud_signal_factor=_number(noise_raw["cloud_signal_factor"], "observation.cloud_signal_factor"),
            lai_saturation=_number(noise_raw["lai_saturation"], "observation.lai_saturation"),
        )
        output = Phase4OutputConfig(
            directory=_resolve_path(source_path.parent, output_raw["directory"]),
            observations_filename=_text(output_raw["observations_filename"], "output.observations_filename"),
            survey_summary_filename=_text(output_raw["survey_summary_filename"], "output.survey_summary_filename"),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ObservationConfigError):
            raise
        raise ObservationConfigError(str(exc)) from exc
    if schedule.end_day >= phase1.calendar.max_duration_days:
        raise ObservationConfigError("survey schedule must fit within the crop campaign")
    issues = load_issue_scenarios(
        issues_file,
        valid_zone_ids=(zone.zone_id for zone in farm.zones),
        campaign_days=phase1.calendar.max_duration_days,
    )
    return Phase4Config(source_path, phase1, phase3, issues_file, issues, schedule, noise, output)


def simulate_observations(
    config: Phase4Config,
    daily_truth: pd.DataFrame,
    *,
    farm: Farm | None = None,
) -> Phase4Result:
    """Generate reproducible weekly observations from crop truth and issues."""

    farm = farm or load_farm(config.phase1)
    truth = _truth_lookup(daily_truth)
    issues_by_zone = {
        zone_id: issue for issue in config.issues for zone_id in issue.zone_ids
    }
    random = Random(config.noise.seed)
    observations: list[Observation] = []
    summaries: list[SurveySummary] = []
    start = config.phase1.calendar.base_sowing_date

    for survey_number, campaign_day in enumerate(config.schedule.survey_days, 1):
        survey_id = f"SURVEY_{survey_number:03d}"
        survey_date = start + timedelta(days=campaign_day)
        mission_config = replace(
            config.phase3,
            survey_date=survey_date,
            grounded=campaign_day in config.schedule.grounded_days,
        )
        mission = plan_mission(mission_config, farm=farm)
        survey_records = _survey_observations(
            survey_id=survey_id,
            survey_date=survey_date,
            campaign_day=campaign_day,
            mission=mission,
            farm=farm,
            truth=truth,
            issues_by_zone=issues_by_zone,
            noise=config.noise,
            random=random,
        )
        observations.extend(survey_records)
        summaries.append(_summarize_survey(survey_id, survey_date, campaign_day, mission, survey_records))
    return Phase4Result(tuple(observations), tuple(summaries))


def _survey_observations(
    *,
    survey_id: str,
    survey_date: date,
    campaign_day: int,
    mission: MissionPlan,
    farm: Farm,
    truth: Mapping[tuple[str, date], Mapping[str, object]],
    issues_by_zone: Mapping[str, IssueScenario],
    noise: SensorNoiseConfig,
    random: Random,
) -> tuple[Observation, ...]:
    clean: dict[str, dict[str, float | bool]] = {}
    for zone in farm.zones:
        row = _truth_row(truth, zone.zone_id, survey_date)
        clean[zone.zone_id] = _clean_signals(
            row,
            issue=issues_by_zone.get(zone.zone_id),
            campaign_day=campaign_day,
            lai_saturation=noise.lai_saturation,
        )

    coverage = {item.zone_id: item for item in mission.coverage}
    illumination = max(0.2, random.gauss(1.0, noise.illumination_std))
    records: list[Observation] = []
    for zone in sorted(farm.zones, key=lambda item: (item.row, item.column)):
        item = coverage[zone.zone_id]
        crop_active = bool(clean[zone.zone_id]["crop_active"])
        if not item.covered:
            records.append(
                Observation(
                    survey_id, survey_date, zone.zone_id, False, 0.0, crop_active,
                    None, None, None, None, None, None, None, None,
                    "grounded" if mission.grounded else "missed_coverage",
                )
            )
            continue

        neighboring = _neighbor_ids(farm, zone.row, zone.column)
        mixed = _mix_signals(clean, zone.zone_id, neighboring, noise.spatial_mix_fraction)
        mixed["nir"] = _mix_value(
            mixed["nir"],
            [float(clean[neighbor]["nir"]) for neighbor in neighboring],
            noise.registration_mix_fraction,
        )
        is_cloudy = random.random() < noise.cloud_probability
        is_missing = random.random() < noise.missing_probability
        if is_missing:
            records.append(
                Observation(
                    survey_id, survey_date, zone.zone_id, True, item.coverage_fraction,
                    crop_active, None, None, None, None, None, None, None, 1.0, "missing",
                )
            )
            continue

        cloud_factor = noise.cloud_signal_factor if is_cloudy else 1.0
        channels: dict[str, float] = {}
        for channel in ("red", "green", "blue", "nir"):
            bias = noise.nir_bias if channel == "nir" else noise.rgb_bias
            value = float(mixed[channel]) * illumination * cloud_factor + bias
            channels[channel] = _clip(value + random.gauss(0.0, noise.random_noise_std))
        denominator = channels["nir"] + channels["red"]
        ndvi = 0.0 if denominator <= 0 else (channels["nir"] - channels["red"]) / denominator
        canopy = _clip(float(mixed["canopy"]) + random.gauss(0.0, noise.random_noise_std))
        uncertainty = _clip(
            noise.random_noise_std * 3
            + noise.spatial_mix_fraction * 0.15
            + noise.registration_mix_fraction * 0.30
            + (0.45 if is_cloudy else 0.0)
        )
        records.append(
            Observation(
                survey_id, survey_date, zone.zone_id, True, item.coverage_fraction,
                crop_active, channels["red"], channels["green"], channels["blue"],
                channels["nir"], ndvi, canopy, None, uncertainty,
                "cloud" if is_cloudy else "good",
            )
        )
    return _add_sensor_only_anomaly_scores(records)


def _clean_signals(
    row: Mapping[str, object],
    *,
    issue: IssueScenario | None,
    campaign_day: int,
    lai_saturation: float,
) -> dict[str, float | bool]:
    crop_active = bool(row["crop_active"])
    lai = max(0.0, _optional_float(row.get("LAI"), 0.0))
    nni = _clip(_optional_float(row.get("NNI"), 1.0))
    sm = _optional_float(row.get("SM"), _optional_float(row.get("soil_smfcf"), 0.3))
    smw = _optional_float(row.get("soil_smw"), 0.1)
    smfcf = _optional_float(row.get("soil_smfcf"), 0.3)
    water = _clip((sm - smw) / max(1e-9, smfcf - smw))
    condition = _clip(0.55 * nni + 0.45 * water)
    canopy = 1.0 - exp(-lai / lai_saturation)
    red = 0.28 * (1 - canopy) + (0.08 + 0.08 * (1 - condition)) * canopy
    green = 0.18 * (1 - canopy) + (0.32 - 0.08 * (1 - condition)) * canopy
    blue = 0.12 * (1 - canopy) + 0.10 * canopy
    nir = 0.22 * (1 - canopy) + (0.72 - 0.20 * (1 - condition)) * canopy

    visibility = issue.visibility_on(campaign_day) if issue is not None else 0.0
    mechanism = issue.mechanism if issue is not None else ""
    effects = {
        "water_deficit": (0.10, -0.12, -0.16, -0.10),
        "excess_water": (0.08, -0.10, -0.12, -0.08),
        "nutrient_deficit": (0.12, -0.15, -0.18, -0.12),
        "canopy_damage": (0.20, -0.18, -0.30, -0.35),
    }
    if mechanism:
        red_effect, green_effect, nir_effect, canopy_effect = effects[mechanism]
        red += red_effect * visibility
        green += green_effect * visibility
        nir += nir_effect * visibility
        canopy += canopy_effect * visibility
    return {
        "crop_active": crop_active,
        "red": _clip(red),
        "green": _clip(green),
        "blue": _clip(blue),
        "nir": _clip(nir),
        "canopy": _clip(canopy),
    }


def _mix_signals(
    signals: Mapping[str, Mapping[str, float | bool]],
    zone_id: str,
    neighbors: tuple[str, ...],
    fraction: float,
) -> dict[str, float]:
    return {
        channel: _mix_value(
            float(signals[zone_id][channel]),
            [float(signals[neighbor][channel]) for neighbor in neighbors],
            fraction,
        )
        for channel in ("red", "green", "blue", "nir", "canopy")
    }


def _mix_value(value: float, neighbors: list[float], fraction: float) -> float:
    if not neighbors:
        return value
    return value * (1.0 - fraction) + sum(neighbors) / len(neighbors) * fraction


def _neighbor_ids(farm: Farm, row: int, column: int) -> tuple[str, ...]:
    positions = {(zone.row, zone.column): zone.zone_id for zone in farm.zones}
    return tuple(
        positions[position]
        for position in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1))
        if position in positions
    )


def _add_sensor_only_anomaly_scores(records: list[Observation]) -> tuple[Observation, ...]:
    valid = [record for record in records if record.ndvi_like is not None]
    if not valid:
        return tuple(records)
    ndvi_reference = median(float(record.ndvi_like) for record in valid)
    green_ratios = [_green_ratio(record) for record in valid]
    green_reference = median(green_ratios)
    scored: list[Observation] = []
    for record in records:
        if record.ndvi_like is None:
            scored.append(record)
            continue
        ndvi_difference = max(0.0, ndvi_reference - record.ndvi_like)
        green_difference = max(0.0, green_reference - _green_ratio(record))
        score = _clip(0.7 * ndvi_difference / 0.25 + 0.3 * green_difference / 0.15)
        scored.append(replace(record, anomaly_score=score))
    return tuple(scored)


def _green_ratio(record: Observation) -> float:
    assert record.relative_red is not None
    assert record.relative_green is not None
    assert record.relative_blue is not None
    total = record.relative_red + record.relative_green + record.relative_blue
    return 0.0 if total <= 0 else record.relative_green / total


def _summarize_survey(
    survey_id: str,
    survey_date: date,
    campaign_day: int,
    mission: MissionPlan,
    observations: tuple[Observation, ...],
) -> SurveySummary:
    return SurveySummary(
        survey_id=survey_id,
        survey_date=survey_date,
        campaign_day=campaign_day,
        grounded=mission.grounded,
        sortie_count=len(mission.sorties),
        covered_zones=sum(item.covered for item in mission.coverage),
        valid_observations=sum(item.ndvi_like is not None for item in observations),
        cloudy_observations=sum(item.quality_flag == "cloud" for item in observations),
        missing_observations=sum(item.quality_flag in {"missing", "grounded", "missed_coverage"} for item in observations),
        total_image_count=sum(sortie.total_image_count for sortie in mission.sorties),
    )


def _truth_lookup(frame: pd.DataFrame) -> dict[tuple[str, date], Mapping[str, object]]:
    required = {"zone_id", "date", "crop_active", "LAI", "NNI", "SM", "soil_smw", "soil_smfcf"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"daily truth is missing columns: {sorted(missing)}")
    lookup: dict[tuple[str, date], Mapping[str, object]] = {}
    for record in frame.to_dict(orient="records"):
        day = record["date"]
        if isinstance(day, datetime):
            day = day.date()
        elif isinstance(day, pd.Timestamp):
            day = day.date()
        elif isinstance(day, str):
            day = date.fromisoformat(day)
        if not isinstance(day, date):
            raise ValueError("daily truth dates must be date-like")
        key = (str(record["zone_id"]), day)
        if key in lookup:
            raise ValueError(f"duplicate daily truth key: {key}")
        lookup[key] = record
    return lookup


def _truth_row(
    truth: Mapping[tuple[str, date], Mapping[str, object]],
    zone_id: str,
    survey_date: date,
) -> Mapping[str, object]:
    exact = truth.get((zone_id, survey_date))
    if exact is not None:
        return exact
    previous = [
        (day, row)
        for (candidate_zone, day), row in truth.items()
        if candidate_zone == zone_id and day < survey_date
    ]
    if not previous:
        raise ValueError(f"daily truth has no row for {zone_id} on or before {survey_date}")
    _, terminal = max(previous, key=lambda item: item[0])
    completed = dict(terminal)
    completed["crop_active"] = False
    return completed


def _optional_float(value: object, default: float) -> float:
    if value is None or pd.isna(value):
        return default
    parsed = float(value)
    return parsed if isfinite(parsed) else default


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ObservationConfigError(f"{context} must be a mapping")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ObservationConfigError(f"{context} must be a list")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = expected - set(mapping)
    extra = set(mapping) - expected
    if missing or extra:
        raise ObservationConfigError(
            f"{context} keys do not match schema; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _resolve_path(base: Path, value: Any) -> Path:
    path = Path(_text(value, "configured path"))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObservationConfigError(f"{name} must be a nonempty string")
    return value.strip()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObservationConfigError(f"{name} must be numeric")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ObservationConfigError(f"{name} must be an integer")
    return value


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _positive_number(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _nonnegative_number(value: float, name: str) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _unit_fraction(value: float, name: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
