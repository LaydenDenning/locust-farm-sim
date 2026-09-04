"""Method-agnostic synthetic crop-monitoring adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
from math import ceil, hypot, radians, sqrt, tan
from random import Random
from typing import Callable, Mapping, Sequence

from src.farm import Farm
from src.simulation.issues import IssueScenario
from src.simulation.profiles import CropProfile, MethodVariant
from src.simulation.routes import AerialLeg, aerial_parallel_sweep, ground_route
from src.simulation.spatial import AnalysisCell


@dataclass(frozen=True)
class CanonicalObservation:
    method_id: str
    method_kind: str
    survey_id: str
    observed_date: date
    available_date: date
    zone_id: str
    footprint: str
    coverage_fraction: float
    measured_features: Mapping[str, float]
    anomaly_score: float | None
    uncertainty: float | None
    quality_flag: str
    provenance: str


@dataclass(frozen=True)
class CanonicalDetection:
    method_id: str
    method_kind: str
    survey_id: str
    survey_date: date
    available_date: date
    zone_id: str
    status: str
    score: float | None
    uncertainty: float | None
    quality_flag: str
    reason: str


@dataclass(frozen=True)
class SurveyUsage:
    method_id: str
    survey_id: str
    campaign_day: int
    completed: bool
    planned_zones: int
    observed_zones: int
    unavailable_zones: int
    duration_hours: float
    sorties: int
    images: int
    scenes: int
    covered_area_ha: float


@dataclass(frozen=True)
class MonitoringResult:
    method_id: str
    method_kind: str
    observations: tuple[CanonicalObservation, ...]
    detections: tuple[CanonicalDetection, ...]
    surveys: tuple[SurveyUsage, ...]
    operating_cost: float


MethodRunner = Callable[..., MonitoringResult]
METHOD_RUNNERS: dict[str, MethodRunner] = {}


def register_method(kind: str, runner: MethodRunner) -> None:
    if not kind.strip() or kind in METHOD_RUNNERS:
        raise ValueError(f"method kind is blank or already registered: {kind!r}")
    METHOD_RUNNERS[kind] = runner


def run_method(
    variant: MethodVariant,
    *,
    farm: Farm,
    cells: tuple[AnalysisCell, ...],
    crop: CropProfile,
    issue: IssueScenario | None,
    affected_fractions: Mapping[str, float],
    scenario_seed: int,
) -> MonitoringResult:
    runner = METHOD_RUNNERS.get(variant.profile.kind)
    if runner is None:
        raise ValueError(f"no runner registered for {variant.profile.kind}")
    return runner(
        variant,
        farm=farm,
        cells=cells,
        crop=crop,
        issue=issue,
        affected_fractions=affected_fractions,
        scenario_seed=scenario_seed,
    )


def _run_aerial(
    variant: MethodVariant,
    *,
    farm: Farm,
    cells: tuple[AnalysisCell, ...],
    crop: CropProfile,
    issue: IssueScenario | None,
    affected_fractions: Mapping[str, float],
    scenario_seed: int,
) -> MonitoringResult:
    settings = variant.profile.settings
    aircraft = _mapping(settings["aircraft"], "aircraft")
    cameras = _sequence(settings["cameras"], "cameras")
    if not cameras:
        raise ValueError("aerial method requires at least one camera")
    altitude = _number(aircraft, "altitude_m")
    speed = _number(aircraft, "mapping_speed_m_s")
    endurance_minutes = _number(aircraft, "usable_endurance_minutes")
    turn_seconds = _number(aircraft, "turn_overhead_seconds")
    takeoff_seconds = _number(aircraft, "takeoff_landing_overhead_seconds")
    battery_seconds = _number(aircraft, "battery_change_seconds")
    footprint_width = min(
        2.0 * altitude * tan(radians(_number(_mapping(camera, "camera"), "horizontal_fov_deg") / 2.0))
        for camera in cameras
    )
    footprint_length = min(
        2.0 * altitude * tan(radians(_number(_mapping(camera, "camera"), "vertical_fov_deg") / 2.0))
        for camera in cameras
    )
    overlap = _number(aircraft, "side_overlap")
    forward_overlap = _number(aircraft, "forward_overlap")
    capture_spacing = footprint_length * (1.0 - forward_overlap)
    if capture_spacing <= 0:
        raise ValueError("forward overlap must be below one")
    route = variant.route
    pattern = str(route.get("pattern", "parallel_sweep"))
    if pattern != "parallel_sweep":
        raise ValueError("aerial methods currently require parallel_sweep")
    targets = tuple(str(item) for item in route.get("target_zone_ids", []))
    legs = aerial_parallel_sweep(
        farm,
        footprint_width_m=footprint_width,
        side_overlap=overlap,
        orientation=str(route.get("orientation", "east_west")),
        target_zone_ids=targets,
    )
    coverage = _aerial_coverage(farm, cells, legs, footprint_width)
    route_distance = sum(leg.distance_m for leg in legs)
    route_distance += sum(
        hypot(second.start_x_m - first.end_x_m, second.start_y_m - first.end_y_m)
        for first, second in zip(legs, legs[1:])
    )
    endurance_seconds = endurance_minutes * 60.0
    flight_seconds = route_distance / speed + max(0, len(legs) - 1) * turn_seconds + takeoff_seconds
    if any(leg.distance_m / speed + takeoff_seconds > endurance_seconds for leg in legs):
        raise ValueError("one aerial route leg does not fit within usable endurance")
    sorties = max(1, ceil(flight_seconds / endurance_seconds))
    duration_hours = (flight_seconds + max(0, sorties - 1) * battery_seconds) / 3600.0
    captures = sum(ceil(leg.distance_m / capture_spacing) + 1 for leg in legs)
    image_count = captures * len(cameras)
    gsd = _number(aircraft, "ground_sample_distance_m")
    delay = int(aircraft.get("processing_delay_days", 0))
    return _simulate_observations(
        variant,
        farm=farm,
        crop=crop,
        issue=issue,
        affected_fractions=affected_fractions,
        scenario_seed=scenario_seed,
        coverage=coverage,
        ground_sample_distance_m=gsd,
        processing_delay_days=delay,
        duration_hours=duration_hours,
        sorties=sorties,
        images=image_count,
        scenes=0,
    )


def _run_ground(
    variant: MethodVariant,
    *,
    farm: Farm,
    cells: tuple[AnalysisCell, ...],
    crop: CropProfile,
    issue: IssueScenario | None,
    affected_fractions: Mapping[str, float],
    scenario_seed: int,
) -> MonitoringResult:
    route = variant.route
    zones = ground_route(
        farm,
        pattern=str(route.get("pattern", "explicit")),
        zone_ids=tuple(str(item) for item in route.get("zone_ids", [])),
        sample_count=int(route["sample_count"]) if "sample_count" in route else None,
        seed=int(route.get("seed", 0)),
        samples_per_leg=int(route.get("samples_per_leg", 3)),
    )
    costs = _mapping(variant.profile.settings["costs"], "costs")
    inspection = _number(costs, "inspection_minutes_per_zone")
    travel = _number(costs, "travel_minutes_between_zones")
    duration = (len(zones) * inspection + max(0, len(zones) - 1) * travel) / 60.0
    coverage = {zone.zone_id: (1.0 if zone.zone_id in zones else 0.0) for zone in farm.zones}
    return _simulate_observations(
        variant,
        farm=farm,
        crop=crop,
        issue=issue,
        affected_fractions=affected_fractions,
        scenario_seed=scenario_seed,
        coverage=coverage,
        ground_sample_distance_m=1.0,
        processing_delay_days=0,
        duration_hours=duration,
        sorties=0,
        images=0,
        scenes=0,
    )


def _run_satellite(
    variant: MethodVariant,
    *,
    farm: Farm,
    cells: tuple[AnalysisCell, ...],
    crop: CropProfile,
    issue: IssueScenario | None,
    affected_fractions: Mapping[str, float],
    scenario_seed: int,
) -> MonitoringResult:
    sensor = _mapping(variant.profile.settings["sensor"], "sensor")
    coverage = {zone.zone_id: 1.0 for zone in farm.zones}
    return _simulate_observations(
        variant,
        farm=farm,
        crop=crop,
        issue=issue,
        affected_fractions=affected_fractions,
        scenario_seed=scenario_seed,
        coverage=coverage,
        ground_sample_distance_m=_number(sensor, "ground_sample_distance_m"),
        processing_delay_days=int(sensor.get("processing_delay_days", 0)),
        duration_hours=0.0,
        sorties=0,
        images=0,
        scenes=1,
    )


def _simulate_observations(
    variant: MethodVariant,
    *,
    farm: Farm,
    crop: CropProfile,
    issue: IssueScenario | None,
    affected_fractions: Mapping[str, float],
    scenario_seed: int,
    coverage: Mapping[str, float],
    ground_sample_distance_m: float,
    processing_delay_days: int,
    duration_hours: float,
    sorties: int,
    images: int,
    scenes: int,
) -> MonitoringResult:
    observation = _mapping(variant.profile.settings["observation"], "observation")
    detection = _mapping(variant.profile.settings["detection"], "detection")
    noise_std = _number(observation, "noise_std")
    sensitivity = _number(observation, "visibility_sensitivity")
    cloud_probability = float(observation.get("cloud_probability", 0.0))
    missing_probability = float(observation.get("missing_probability", 0.0))
    threshold = _number(detection, "anomaly_threshold")
    maximum_uncertainty = _number(detection, "maximum_uncertainty")
    allowed = set(str(value) for value in _sequence(detection["allowed_quality_flags"], "allowed_quality_flags"))
    sowing = crop.calendar.base_sowing_date
    observations: list[CanonicalObservation] = []
    detections: list[CanonicalDetection] = []
    surveys: list[SurveyUsage] = []
    completed_count = 0
    for survey_number, campaign_day in enumerate(variant.schedule.days, start=1):
        survey_id = f"SURVEY_{survey_number:03d}"
        survey_date = sowing + timedelta(days=campaign_day)
        family_random = _uniform(scenario_seed, variant.profile.family, campaign_day, "mission")
        missed = campaign_day in variant.forced_missed_days or family_random < variant.missed_probability
        completed = not missed
        if completed:
            completed_count += 1
        survey_records: list[CanonicalObservation] = []
        for zone in sorted(farm.zones, key=lambda item: (item.row, item.column)):
            zone_coverage = 0.0 if missed else float(coverage.get(zone.zone_id, 0.0))
            quality = "good"
            score: float | None
            uncertainty: float | None
            if zone_coverage <= 0:
                quality = "missed_survey" if missed else "not_observed"
                score = None
                uncertainty = None
                features: Mapping[str, float] = {}
            elif _uniform(scenario_seed, variant.profile.family, campaign_day, zone.zone_id, "missing") < missing_probability:
                quality = "missing"
                score = None
                uncertainty = 1.0
                features = {}
            elif _uniform(scenario_seed, variant.profile.family, campaign_day, zone.zone_id, "cloud") < cloud_probability:
                quality = "cloud"
                score = None
                uncertainty = 1.0
                features = {}
            else:
                fraction = affected_fractions.get(zone.zone_id, 0.0)
                visibility = issue.visibility_on(campaign_day) if issue is not None and zone.zone_id in issue.zone_ids else 0.0
                affected_area = zone.area_m2 * fraction
                resolution_factor = min(1.0, sqrt(max(affected_area, 0.0)) / max(ground_sample_distance_m, 1e-9))
                noise = Random(_stable_seed(scenario_seed, variant.profile.family, campaign_day, zone.zone_id, "noise")).gauss(0.0, noise_std)
                score = _clip(crop.observation.healthy_visual_baseline + sensitivity * visibility * resolution_factor + noise)
                uncertainty = _clip(noise_std * 3.0 + (1.0 - zone_coverage) * 0.25)
                red_effect, green_effect, nir_effect, canopy_effect = crop.observation.reflectance_effects[issue.mechanism] if issue is not None and zone.zone_id in issue.zone_ids else (0.0, 0.0, 0.0, 0.0)
                signal = visibility * resolution_factor
                features = {
                    "red": _clip(0.10 + red_effect * signal),
                    "green": _clip(0.30 + green_effect * signal),
                    "blue": 0.10,
                    "nir": _clip(0.70 + nir_effect * signal),
                    "canopy": _clip(0.80 + canopy_effect * signal),
                }
            available_date = survey_date + timedelta(days=processing_delay_days)
            record = CanonicalObservation(variant.method_id, variant.profile.kind, survey_id, survey_date, available_date, zone.zone_id, zone.zone_id, zone_coverage, features, score, uncertainty, quality, f"synthetic:{variant.profile.source_path.name}")
            survey_records.append(record)
            observations.append(record)
            status, reason = _classify(record, threshold, maximum_uncertainty, allowed)
            detections.append(CanonicalDetection(variant.method_id, variant.profile.kind, survey_id, survey_date, available_date, zone.zone_id, status, score, uncertainty, quality, reason))
        observed = sum(item.anomaly_score is not None for item in survey_records)
        surveys.append(SurveyUsage(variant.method_id, survey_id, campaign_day, completed, len(farm.zones), observed, len(farm.zones) - observed, duration_hours if completed else 0.0, sorties if completed else 0, images if completed else 0, scenes if completed else 0, sum(zone.area_m2 * coverage.get(zone.zone_id, 0.0) for zone in farm.zones) / 10_000.0 if completed else 0.0))
    cost = _operating_cost(variant, tuple(surveys), completed_count)
    return MonitoringResult(variant.method_id, variant.profile.kind, tuple(observations), tuple(detections), tuple(surveys), cost)


def _classify(record: CanonicalObservation, threshold: float, maximum_uncertainty: float, allowed: set[str]) -> tuple[str, str]:
    if record.anomaly_score is None or record.uncertainty is None:
        return "unavailable", f"quality_{record.quality_flag}"
    if record.quality_flag not in allowed:
        return "unavailable", f"quality_{record.quality_flag}"
    if record.uncertainty > maximum_uncertainty:
        return "unavailable", "uncertainty_above_limit"
    if record.anomaly_score >= threshold:
        return "flagged", "anomaly_threshold_met"
    return "clear", "anomaly_below_threshold"


def _operating_cost(variant: MethodVariant, surveys: tuple[SurveyUsage, ...], completed_count: int) -> float:
    costs = _mapping(variant.profile.settings["costs"], "costs")
    kind = variant.profile.kind
    if kind == "drone":
        return sum(item.sorties for item in surveys) * float(costs["flight_cost_per_sortie"]) + sum(item.images for item in surveys) / 1000.0 * float(costs["processing_cost_per_1000_images"])
    if kind == "ground_scout":
        return sum(item.duration_hours for item in surveys) * float(costs["labor_cost_per_hour"])
    if kind == "manned_aircraft":
        return (float(costs["mobilization_cost"]) if completed_count else 0.0) + sum(item.duration_hours for item in surveys) * float(costs["flight_cost_per_hour"]) + sum(item.covered_area_ha for item in surveys) * float(costs["processing_cost_per_ha"])
    if kind == "satellite":
        return sum(item.scenes for item in surveys) * (float(costs["data_cost_per_scene"]) + float(costs["processing_cost_per_scene"]))
    raise ValueError(f"unsupported cost model: {kind}")


def _aerial_coverage(farm: Farm, cells: tuple[AnalysisCell, ...], legs: tuple[AerialLeg, ...], width: float) -> Mapping[str, float]:
    half = width / 2.0
    totals: dict[str, int] = {}
    hits: dict[str, int] = {}
    for cell in cells:
        totals[cell.zone_id] = totals.get(cell.zone_id, 0) + 1
        if any(_point_near_leg(cell.center_x_m, cell.center_y_m, leg, half) for leg in legs):
            hits[cell.zone_id] = hits.get(cell.zone_id, 0) + 1
    return {zone.zone_id: hits.get(zone.zone_id, 0) / totals[zone.zone_id] for zone in farm.zones}


def _point_near_leg(x: float, y: float, leg: AerialLeg, half_width: float) -> bool:
    if leg.start_y_m == leg.end_y_m:
        return min(leg.start_x_m, leg.end_x_m) <= x <= max(leg.start_x_m, leg.end_x_m) and abs(y - leg.start_y_m) <= half_width
    return min(leg.start_y_m, leg.end_y_m) <= y <= max(leg.start_y_m, leg.end_y_m) and abs(x - leg.start_x_m) <= half_width


def _stable_seed(*parts: object) -> int:
    digest = sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _uniform(*parts: object) -> float:
    return Random(_stable_seed(*parts)).random()


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    return value


def _number(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


register_method("drone", _run_aerial)
register_method("manned_aircraft", _run_aerial)
register_method("ground_scout", _run_ground)
register_method("satellite", _run_satellite)
