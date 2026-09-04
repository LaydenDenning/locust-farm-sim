"""Configuration models for modular crop-monitoring experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from math import isfinite, radians, tan
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.farm import Phase1Config, load_phase1_config
from src.farm.config import CalendarConfig, CropConfig
from src.simulation.issues import ISSUE_MECHANISMS, InterventionRule


class ExperimentConfigError(ValueError):
    """Raised when an experiment or referenced profile is invalid."""


@dataclass(frozen=True)
class NumberRange:
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not isfinite(self.minimum) or not isfinite(self.maximum) or self.minimum > self.maximum:
            raise ValueError("numeric range must contain finite ordered bounds")


@dataclass(frozen=True)
class IntegerRange:
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum < 0 or self.minimum > self.maximum:
            raise ValueError("integer range must contain nonnegative ordered bounds")


@dataclass(frozen=True)
class ScheduleConfig:
    days: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.days:
            raise ValueError("schedule must contain at least one day")
        if len(set(self.days)) != len(self.days) or tuple(sorted(self.days)) != self.days:
            raise ValueError("schedule days must be sorted and unique")
        if any(not isinstance(day, int) or isinstance(day, bool) or day < 0 for day in self.days):
            raise ValueError("schedule days must be nonnegative integers")


@dataclass(frozen=True)
class CropObservationProfile:
    lai_saturation: float
    healthy_visual_baseline: float
    reflectance_effects: Mapping[str, tuple[float, float, float, float]]


@dataclass(frozen=True)
class CropProfile:
    source_path: Path
    crop: CropConfig
    calendar: CalendarConfig
    observation: CropObservationProfile
    required_truth_variables: tuple[str, ...]
    twso_proxy_value_per_tonne: float


@dataclass(frozen=True)
class ScenarioProfile:
    source_path: Path
    no_issue_probability: float
    footprint_zone_count: IntegerRange
    footprint_cell_count: IntegerRange
    onset_day: IntegerRange
    progression_per_day: NumberRange
    max_severity: NumberRange
    visibility_delay_days: IntegerRange
    visibility_scale: NumberRange
    untreated_loss_fraction: NumberRange
    efficacy_multiplier: NumberRange
    response_delay_days: IntegerRange
    treatment_cost_multiplier: NumberRange
    twso_value_multiplier: NumberRange
    spatial_footprints: bool
    analysis_cell_size_m: float


@dataclass(frozen=True)
class MethodProfile:
    source_path: Path
    kind: str
    family: str
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class MethodVariant:
    method_id: str
    profile: MethodProfile
    schedule: ScheduleConfig
    route: Mapping[str, Any]
    missed_probability: float
    forced_missed_days: tuple[int, ...]
    confirmation_delay_days: int


@dataclass(frozen=True)
class OutputProfile:
    directory: Path
    method_results_filename: str
    pairwise_results_filename: str
    distribution_summary_filename: str
    sensitivity_filename: str
    manifest_filename: str


@dataclass(frozen=True)
class ExperimentConfig:
    source_path: Path
    phase1: Phase1Config
    crop_profile: CropProfile
    scenario_profile: ScenarioProfile
    intervention_rules: tuple[InterventionRule, ...]
    economics: Mapping[str, Any]
    methods: tuple[MethodVariant, ...]
    reference_method_id: str
    scenario_count: int
    seed: int
    compatibility_phase11_config: Path | None
    output: OutputProfile
    input_paths: tuple[Path, ...]


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    root = _yaml_mapping(source)
    _exact(root, {"farm_config", "crop_profile", "scenario_profile", "intervention_profile", "economics_profile", "scenario_count", "seed", "reference_method_id", "methods", "compatibility_phase11_config", "output"}, "experiment")
    phase1_path = _resolve(source.parent, root["farm_config"])
    crop_path = _resolve(source.parent, root["crop_profile"])
    scenario_path = _resolve(source.parent, root["scenario_profile"])
    intervention_path = _resolve(source.parent, root["intervention_profile"])
    economics_path = _resolve(source.parent, root["economics_profile"])
    base = load_phase1_config(phase1_path)
    crop_profile = load_crop_profile(crop_path)
    phase1 = replace(base, crop=crop_profile.crop, calendar=crop_profile.calendar)
    scenario = load_scenario_profile(scenario_path)
    rules = load_intervention_profile(intervention_path)
    economics = load_economics_profile(economics_path)

    method_rows = _list(root["methods"], "methods")
    methods: list[MethodVariant] = []
    method_paths: list[Path] = []
    for index, value in enumerate(method_rows, start=1):
        context = f"methods[{index}]"
        row = _mapping(value, context)
        _exact(row, {"id", "profile", "schedule", "route", "reliability", "confirmation_delay_days"}, context)
        profile_path = _resolve(source.parent, row["profile"])
        method_paths.append(profile_path)
        reliability = _mapping(row["reliability"], f"{context}.reliability")
        _exact(reliability, {"missed_probability", "forced_missed_days"}, f"{context}.reliability")
        missed_probability = _number(reliability["missed_probability"], f"{context}.reliability.missed_probability")
        if not 0 <= missed_probability <= 1:
            raise ExperimentConfigError(f"{context}.reliability.missed_probability must be between 0 and 1")
        schedule = _schedule(row["schedule"], f"{context}.schedule")
        forced = tuple(_integer(item, f"{context}.forced_missed_days") for item in _list(reliability["forced_missed_days"], f"{context}.forced_missed_days"))
        if set(forced) - set(schedule.days):
            raise ExperimentConfigError(f"{context} forced missed days must be scheduled")
        methods.append(MethodVariant(
            method_id=_text(row["id"], f"{context}.id"),
            profile=load_method_profile(profile_path),
            schedule=schedule,
            route=dict(_mapping(row["route"], f"{context}.route")),
            missed_probability=missed_probability,
            forced_missed_days=forced,
            confirmation_delay_days=_integer(row["confirmation_delay_days"], f"{context}.confirmation_delay_days"),
        ))
    ids = [method.method_id for method in methods]
    if len(ids) != len(set(ids)):
        raise ExperimentConfigError("method IDs must be unique")
    reference = _text(root["reference_method_id"], "reference_method_id")
    if reference not in ids:
        raise ExperimentConfigError("reference_method_id must identify a configured method")
    if next(item for item in methods if item.method_id == reference).profile.kind != "ground_scout":
        raise ExperimentConfigError("reference_method_id must identify a ground scout")
    count = _integer(root["scenario_count"], "scenario_count")
    if count < 1:
        raise ExperimentConfigError("scenario_count must be positive")
    seed = _integer(root["seed"], "seed")
    compatibility_path = None if root["compatibility_phase11_config"] is None else _resolve(source.parent, root["compatibility_phase11_config"])
    if max(day for method in methods for day in method.schedule.days) >= phase1.calendar.max_duration_days:
        raise ExperimentConfigError("method schedules must fit within the crop campaign")

    output_raw = _mapping(root["output"], "output")
    _exact(output_raw, {"directory", "method_results_filename", "pairwise_results_filename", "distribution_summary_filename", "sensitivity_filename", "manifest_filename"}, "output")
    output = OutputProfile(
        directory=_resolve(source.parent, output_raw["directory"]),
        method_results_filename=_filename(output_raw["method_results_filename"], "output.method_results_filename"),
        pairwise_results_filename=_filename(output_raw["pairwise_results_filename"], "output.pairwise_results_filename"),
        distribution_summary_filename=_filename(output_raw["distribution_summary_filename"], "output.distribution_summary_filename"),
        sensitivity_filename=_filename(output_raw["sensitivity_filename"], "output.sensitivity_filename"),
        manifest_filename=_filename(output_raw["manifest_filename"], "output.manifest_filename"),
    )
    inputs = (source, phase1_path, crop_path, scenario_path, intervention_path, economics_path, *method_paths, *((compatibility_path,) if compatibility_path else ()))
    return ExperimentConfig(source, phase1, crop_profile, scenario, rules, economics, tuple(methods), reference, count, seed, compatibility_path, output, tuple(inputs))


def load_crop_profile(path: str | Path) -> CropProfile:
    source = Path(path).resolve()
    root = _yaml_mapping(source)
    _exact(root, {"crop", "calendar", "observation", "required_truth_variables", "twso_proxy_value_per_tonne"}, "crop profile")
    crop = _mapping(root["crop"], "crop")
    _exact(crop, {"parameter_directory", "crop_name", "variety_name", "model_name"}, "crop")
    calendar = _mapping(root["calendar"], "calendar")
    _exact(calendar, {"base_sowing_date", "crop_start_type", "crop_end_type", "max_duration_days"}, "calendar")
    observation = _mapping(root["observation"], "observation")
    _exact(observation, {"lai_saturation", "healthy_visual_baseline", "reflectance_effects"}, "observation")
    effects_raw = _mapping(observation["reflectance_effects"], "observation.reflectance_effects")
    _exact(effects_raw, set(ISSUE_MECHANISMS), "observation.reflectance_effects")
    effects: dict[str, tuple[float, float, float, float]] = {}
    for mechanism, values in effects_raw.items():
        numbers = tuple(_number(item, f"effect {mechanism}") for item in _list(values, f"effect {mechanism}"))
        if len(numbers) != 4:
            raise ExperimentConfigError(f"effect {mechanism} requires red, green, nir, canopy values")
        effects[mechanism] = numbers
    crop_config = CropConfig(
        _resolve(source.parent, crop["parameter_directory"]),
        _text(crop["crop_name"], "crop.crop_name"),
        _text(crop["variety_name"], "crop.variety_name"),
        _text(crop["model_name"], "crop.model_name"),
    )
    calendar_config = CalendarConfig(
        _date(calendar["base_sowing_date"], "calendar.base_sowing_date"),
        _text(calendar["crop_start_type"], "calendar.crop_start_type"),
        _text(calendar["crop_end_type"], "calendar.crop_end_type"),
        _integer(calendar["max_duration_days"], "calendar.max_duration_days"),
    )
    required = tuple(_text(item, "required truth variable") for item in _list(root["required_truth_variables"], "required_truth_variables"))
    if not required or len(required) != len(set(required)):
        raise ExperimentConfigError("required_truth_variables must be nonempty and unique")
    value = _number(root["twso_proxy_value_per_tonne"], "twso_proxy_value_per_tonne")
    if value < 0:
        raise ExperimentConfigError("twso_proxy_value_per_tonne must be nonnegative")
    return CropProfile(source, crop_config, calendar_config, CropObservationProfile(_number(observation["lai_saturation"], "observation.lai_saturation"), _number(observation["healthy_visual_baseline"], "observation.healthy_visual_baseline"), effects), required, value)


def load_scenario_profile(path: str | Path) -> ScenarioProfile:
    source = Path(path).resolve()
    root = _yaml_mapping(source)
    expected = {"no_issue_probability", "footprint_zone_count", "footprint_cell_count", "onset_day", "progression_per_day", "max_severity", "visibility_delay_days", "visibility_scale", "untreated_loss_fraction", "efficacy_multiplier", "response_delay_days", "treatment_cost_multiplier", "twso_value_multiplier", "spatial_footprints", "analysis_cell_size_m"}
    _exact(root, expected, "scenario profile")
    probability = _number(root["no_issue_probability"], "no_issue_probability")
    if not 0 <= probability <= 1:
        raise ExperimentConfigError("no_issue_probability must be between 0 and 1")
    return ScenarioProfile(source, probability, _integer_range(root["footprint_zone_count"], "footprint_zone_count"), _integer_range(root["footprint_cell_count"], "footprint_cell_count"), _integer_range(root["onset_day"], "onset_day"), _number_range(root["progression_per_day"], "progression_per_day"), _number_range(root["max_severity"], "max_severity"), _integer_range(root["visibility_delay_days"], "visibility_delay_days"), _number_range(root["visibility_scale"], "visibility_scale"), _number_range(root["untreated_loss_fraction"], "untreated_loss_fraction"), _number_range(root["efficacy_multiplier"], "efficacy_multiplier"), _integer_range(root["response_delay_days"], "response_delay_days"), _number_range(root["treatment_cost_multiplier"], "treatment_cost_multiplier"), _number_range(root["twso_value_multiplier"], "twso_value_multiplier"), _boolean(root["spatial_footprints"], "spatial_footprints"), _number(root["analysis_cell_size_m"], "analysis_cell_size_m"))


def load_intervention_profile(path: str | Path) -> tuple[InterventionRule, ...]:
    source = Path(path).resolve()
    root = _yaml_mapping(source)
    _exact(root, set(ISSUE_MECHANISMS), "intervention profile")
    rules = []
    for mechanism in ISSUE_MECHANISMS:
        values = _mapping(root[mechanism], mechanism)
        _exact(values, {"response_delay_days", "efficacy", "cutoff_day", "cost_per_ha"}, mechanism)
        rules.append(InterventionRule(mechanism, _integer(values["response_delay_days"], f"{mechanism}.response_delay_days"), _number(values["efficacy"], f"{mechanism}.efficacy"), _integer(values["cutoff_day"], f"{mechanism}.cutoff_day"), _number(values["cost_per_ha"], f"{mechanism}.cost_per_ha")))
    return tuple(rules)


def load_economics_profile(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).resolve()
    root = _yaml_mapping(source)
    _exact(root, {"currency", "confirmation_cost_per_visit", "false_positive_action_cost_per_ha", "confirmation_sensitivity", "confirmation_specificity"}, "economics profile")
    values = {"currency": _text(root["currency"], "currency"), "confirmation_cost_per_visit": _number(root["confirmation_cost_per_visit"], "confirmation_cost_per_visit"), "false_positive_action_cost_per_ha": _number(root["false_positive_action_cost_per_ha"], "false_positive_action_cost_per_ha"), "confirmation_sensitivity": _number(root["confirmation_sensitivity"], "confirmation_sensitivity"), "confirmation_specificity": _number(root["confirmation_specificity"], "confirmation_specificity")}
    if values["confirmation_cost_per_visit"] < 0 or values["false_positive_action_cost_per_ha"] < 0:
        raise ExperimentConfigError("economic costs must be nonnegative")
    if not 0 <= values["confirmation_sensitivity"] <= 1 or not 0 <= values["confirmation_specificity"] <= 1:
        raise ExperimentConfigError("confirmation sensitivity and specificity must be between 0 and 1")
    return values


def load_method_profile(path: str | Path) -> MethodProfile:
    source = Path(path).resolve()
    root = _yaml_mapping(source)
    _exact(root, {"kind", "family", "settings"}, "method profile")
    kind = _text(root["kind"], "kind")
    if kind not in {"drone", "ground_scout", "manned_aircraft", "satellite"}:
        raise ExperimentConfigError(f"unsupported method kind: {kind}")
    settings = dict(_mapping(root["settings"], "settings"))
    required = {
        "drone": {"aircraft", "cameras", "observation", "detection", "costs"},
        "manned_aircraft": {"aircraft", "cameras", "observation", "detection", "costs"},
        "ground_scout": {"observation", "detection", "costs"},
        "satellite": {"sensor", "observation", "detection", "costs"},
    }[kind]
    _exact(settings, required, "settings")
    _validate_method_settings(kind, settings)
    return MethodProfile(source, kind, _text(root["family"], "family"), settings)


def _validate_method_settings(kind: str, settings: Mapping[str, Any]) -> None:
    observation = _mapping(settings["observation"], "settings.observation")
    _exact(observation, {"noise_std", "visibility_sensitivity", "cloud_probability", "missing_probability"}, "settings.observation")
    if _number(observation["noise_std"], "observation.noise_std") < 0:
        raise ExperimentConfigError("observation.noise_std must be nonnegative")
    for key in ("visibility_sensitivity", "cloud_probability", "missing_probability"):
        value = _number(observation[key], f"observation.{key}")
        if not 0 <= value <= 1:
            raise ExperimentConfigError(f"observation.{key} must be between 0 and 1")
    detection = _mapping(settings["detection"], "settings.detection")
    _exact(detection, {"anomaly_threshold", "maximum_uncertainty", "allowed_quality_flags"}, "settings.detection")
    for key in ("anomaly_threshold", "maximum_uncertainty"):
        value = _number(detection[key], f"detection.{key}")
        if not 0 <= value <= 1:
            raise ExperimentConfigError(f"detection.{key} must be between 0 and 1")
    flags = [_text(item, "allowed quality flag") for item in _list(detection["allowed_quality_flags"], "allowed_quality_flags")]
    if not flags or len(flags) != len(set(flags)):
        raise ExperimentConfigError("allowed_quality_flags must be nonempty and unique")

    costs = _mapping(settings["costs"], "settings.costs")
    cost_keys = {
        "drone": {"flight_cost_per_sortie", "processing_cost_per_1000_images"},
        "ground_scout": {"inspection_minutes_per_zone", "travel_minutes_between_zones", "labor_cost_per_hour"},
        "manned_aircraft": {"mobilization_cost", "flight_cost_per_hour", "processing_cost_per_ha"},
        "satellite": {"data_cost_per_scene", "processing_cost_per_scene"},
    }[kind]
    _exact(costs, cost_keys, "settings.costs")
    if any(_number(costs[key], f"costs.{key}") < 0 for key in cost_keys):
        raise ExperimentConfigError("method costs must be nonnegative")

    if kind in {"drone", "manned_aircraft"}:
        aircraft = _mapping(settings["aircraft"], "settings.aircraft")
        aircraft_keys = {"name", "mapping_speed_m_s", "usable_endurance_minutes", "altitude_m", "ground_sample_distance_m", "forward_overlap", "side_overlap", "turn_overhead_seconds", "takeoff_landing_overhead_seconds", "battery_change_seconds", "processing_delay_days"}
        _exact(aircraft, aircraft_keys, "settings.aircraft")
        _text(aircraft["name"], "aircraft.name")
        for key in ("mapping_speed_m_s", "usable_endurance_minutes", "altitude_m", "ground_sample_distance_m"):
            if _number(aircraft[key], f"aircraft.{key}") <= 0:
                raise ExperimentConfigError(f"aircraft.{key} must be positive")
        for key in ("forward_overlap", "side_overlap"):
            value = _number(aircraft[key], f"aircraft.{key}")
            if not 0 <= value < 1:
                raise ExperimentConfigError(f"aircraft.{key} must be at least zero and below one")
        for key in ("turn_overhead_seconds", "takeoff_landing_overhead_seconds", "battery_change_seconds"):
            if _number(aircraft[key], f"aircraft.{key}") < 0:
                raise ExperimentConfigError(f"aircraft.{key} must be nonnegative")
        if _integer(aircraft["processing_delay_days"], "aircraft.processing_delay_days") < 0:
            raise ExperimentConfigError("processing delay must be nonnegative")
        cameras = _list(settings["cameras"], "settings.cameras")
        if not cameras:
            raise ExperimentConfigError("aerial method requires cameras")
        names = []
        capture_intervals = []
        ground_lengths = []
        for index, item in enumerate(cameras, start=1):
            camera = _mapping(item, f"camera {index}")
            _exact(camera, {"name", "channel", "horizontal_fov_deg", "vertical_fov_deg", "minimum_capture_interval_seconds"}, f"camera {index}")
            names.append(_text(camera["name"], f"camera {index}.name"))
            _text(camera["channel"], f"camera {index}.channel")
            for key in ("horizontal_fov_deg", "vertical_fov_deg"):
                value = _number(camera[key], f"camera {index}.{key}")
                if not 0 < value < 180:
                    raise ExperimentConfigError(f"camera {index}.{key} must be between 0 and 180")
            capture = _number(camera["minimum_capture_interval_seconds"], f"camera {index}.minimum_capture_interval_seconds")
            if capture <= 0:
                raise ExperimentConfigError("camera capture interval must be positive")
            capture_intervals.append(capture)
            ground_lengths.append(2 * float(aircraft["altitude_m"]) * tan(radians(float(camera["vertical_fov_deg"]) / 2)))
        if len(names) != len(set(names)):
            raise ExperimentConfigError("camera names must be unique")
        required_interval = min(ground_lengths) * (1 - float(aircraft["forward_overlap"])) / float(aircraft["mapping_speed_m_s"])
        if required_interval < max(capture_intervals):
            raise ExperimentConfigError("mapping speed and overlap require captures faster than a camera supports")
    elif kind == "satellite":
        sensor = _mapping(settings["sensor"], "settings.sensor")
        _exact(sensor, {"name", "ground_sample_distance_m", "bands", "processing_delay_days"}, "settings.sensor")
        _text(sensor["name"], "sensor.name")
        if _number(sensor["ground_sample_distance_m"], "sensor.ground_sample_distance_m") <= 0:
            raise ExperimentConfigError("satellite ground sample distance must be positive")
        bands = [_text(item, "sensor band") for item in _list(sensor["bands"], "sensor.bands")]
        if not bands or len(bands) != len(set(bands)):
            raise ExperimentConfigError("sensor bands must be nonempty and unique")
        if _integer(sensor["processing_delay_days"], "sensor.processing_delay_days") < 0:
            raise ExperimentConfigError("processing delay must be nonnegative")


def _schedule(value: Any, context: str) -> ScheduleConfig:
    row = _mapping(value, context)
    if set(row) == {"days"}:
        days = tuple(_integer(item, f"{context}.days") for item in _list(row["days"], f"{context}.days"))
    elif set(row) == {"start_day", "end_day", "interval_days"}:
        start = _integer(row["start_day"], f"{context}.start_day")
        end = _integer(row["end_day"], f"{context}.end_day")
        interval = _integer(row["interval_days"], f"{context}.interval_days")
        if interval < 1 or end < start:
            raise ExperimentConfigError(f"{context} interval schedule is invalid")
        days = tuple(range(start, end + 1, interval))
    else:
        raise ExperimentConfigError(f"{context} must use days or start/end/interval")
    return ScheduleConfig(days)


def _yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ExperimentConfigError(f"unable to load {path}: {exc}") from exc
    return _mapping(raw, str(path))


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentConfigError(f"{context} must be a mapping")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExperimentConfigError(f"{context} must be a list")
    return value


def _exact(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing, extra = expected - set(value), set(value) - expected
    if missing or extra:
        raise ExperimentConfigError(f"{context} keys do not match schema; missing={sorted(missing)}, extra={sorted(extra)}")


def _resolve(parent: Path, value: Any) -> Path:
    path = Path(_text(value, "path"))
    return (parent / path).resolve() if not path.is_absolute() else path.resolve()


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigError(f"{context} must be nonempty text")
    return value.strip()


def _filename(value: Any, context: str) -> str:
    text = _text(value, context)
    if Path(text).name != text:
        raise ExperimentConfigError(f"{context} must be a simple filename")
    return text


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentConfigError(f"{context} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ExperimentConfigError(f"{context} must be finite")
    return result


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExperimentConfigError(f"{context} must be an integer")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ExperimentConfigError(f"{context} must be boolean")
    return value


def _date(value: Any, context: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(_text(value, context))
    except ValueError as exc:
        raise ExperimentConfigError(f"{context} must be an ISO date") from exc


def _number_range(value: Any, context: str) -> NumberRange:
    rows = _list(value, context)
    if len(rows) != 2:
        raise ExperimentConfigError(f"{context} requires two bounds")
    return NumberRange(_number(rows[0], context), _number(rows[1], context))


def _integer_range(value: Any, context: str) -> IntegerRange:
    rows = _list(value, context)
    if len(rows) != 2:
        raise ExperimentConfigError(f"{context} requires two bounds")
    return IntegerRange(_integer(rows[0], context), _integer(rows[1], context))
