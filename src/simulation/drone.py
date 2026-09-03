"""Deterministic Phase 3 drone mission planning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from math import ceil, isfinite, radians, tan
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.farm import Farm, Phase1Config, load_farm, load_phase1_config


class DroneConfigError(ValueError):
    """Raised when Phase 3 configuration is invalid."""


@dataclass(frozen=True)
class CameraConfig:
    name: str
    channel: str
    width_px: int
    height_px: int
    horizontal_fov_deg: float
    vertical_fov_deg: float
    minimum_capture_interval_seconds: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("camera name must not be empty")
        if self.channel not in {"rgb", "nir"}:
            raise ValueError(f"{self.name}: channel must be 'rgb' or 'nir'")
        _positive_integer(self.width_px, f"{self.name}: width_px")
        _positive_integer(self.height_px, f"{self.name}: height_px")
        for name in ("horizontal_fov_deg", "vertical_fov_deg"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 < value < 180:
                raise ValueError(f"{self.name}: {name} must be between 0 and 180")
        _positive_number(
            self.minimum_capture_interval_seconds,
            f"{self.name}: minimum_capture_interval_seconds",
        )

    def ground_width_m(self, altitude_m: float) -> float:
        return 2.0 * altitude_m * tan(radians(self.horizontal_fov_deg / 2.0))

    def ground_length_m(self, altitude_m: float) -> float:
        return 2.0 * altitude_m * tan(radians(self.vertical_fov_deg / 2.0))


@dataclass(frozen=True)
class DroneConfig:
    aircraft_name: str
    reference_mass_g: float
    maximum_flight_time_minutes: float
    maximum_speed_m_s: float
    mapping_speed_m_s: float
    reserve_fraction: float
    payload_endurance_factor: float
    altitude_m: float
    forward_overlap: float
    side_overlap: float
    turn_overhead_seconds: float
    takeoff_landing_overhead_seconds: float
    battery_change_seconds: float

    def __post_init__(self) -> None:
        if not self.aircraft_name.strip():
            raise ValueError("aircraft_name must not be empty")
        for name in (
            "reference_mass_g",
            "maximum_flight_time_minutes",
            "maximum_speed_m_s",
            "mapping_speed_m_s",
            "payload_endurance_factor",
            "altitude_m",
        ):
            _positive_number(getattr(self, name), name)
        for name in (
            "turn_overhead_seconds",
            "takeoff_landing_overhead_seconds",
            "battery_change_seconds",
        ):
            _nonnegative_number(getattr(self, name), name)
        _fraction_below_one(self.reserve_fraction, "reserve_fraction")
        _fraction_below_one(self.forward_overlap, "forward_overlap")
        _fraction_below_one(self.side_overlap, "side_overlap")
        if self.payload_endurance_factor > 1:
            raise ValueError("payload_endurance_factor must be at most 1")
        if self.mapping_speed_m_s > self.maximum_speed_m_s:
            raise ValueError("mapping speed must not exceed maximum speed")

    @property
    def usable_endurance_seconds(self) -> float:
        return (
            self.maximum_flight_time_minutes
            * 60.0
            * (1.0 - self.reserve_fraction)
            * self.payload_endurance_factor
        )


@dataclass(frozen=True)
class Phase3OutputConfig:
    directory: Path
    flight_lines_filename: str
    sortie_summary_filename: str
    zone_coverage_filename: str

    def __post_init__(self) -> None:
        for name in (
            "flight_lines_filename",
            "sortie_summary_filename",
            "zone_coverage_filename",
        ):
            value = getattr(self, name)
            if not value.strip() or Path(value).name != value:
                raise ValueError(f"output.{name} must be a simple file name")


@dataclass(frozen=True)
class Phase3Config:
    source_path: Path
    phase1: Phase1Config
    survey_date: date
    start_time: time
    grounded: bool
    drone: DroneConfig
    cameras: tuple[CameraConfig, ...]
    output: Phase3OutputConfig

    def __post_init__(self) -> None:
        if not isinstance(self.grounded, bool):
            raise ValueError("survey.grounded must be boolean")
        if len(self.cameras) != 2:
            raise ValueError("Phase 3 requires exactly one RGB and one NIR camera")
        channels = [camera.channel for camera in self.cameras]
        if set(channels) != {"rgb", "nir"} or len(channels) != len(set(channels)):
            raise ValueError("Phase 3 requires exactly one RGB and one NIR camera")
        names = [camera.name for camera in self.cameras]
        if len(names) != len(set(names)):
            raise ValueError("camera names must be unique")


@dataclass(frozen=True)
class FlightLine:
    line_id: str
    sortie_id: str
    sequence: int
    y_m: float
    start_x_m: float
    end_x_m: float
    distance_m: float
    direction: str
    capture_count_per_camera: int
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class SortieSummary:
    sortie_id: str
    line_count: int
    route_distance_m: float
    flight_time_seconds: float
    usable_endurance_seconds: float
    battery_margin_seconds: float
    capture_count_per_camera: int
    total_image_count: int


@dataclass(frozen=True)
class ZoneCoverage:
    zone_id: str
    covered: bool
    coverage_fraction: float
    line_ids: tuple[str, ...]
    first_observation_seconds: float | None
    first_observation_time: datetime | None


@dataclass(frozen=True)
class MissionPlan:
    survey_date: date
    grounded: bool
    footprint_width_m: float
    footprint_length_m: float
    line_spacing_m: float
    capture_spacing_m: float
    capture_interval_seconds: float
    usable_endurance_seconds: float
    lines: tuple[FlightLine, ...]
    sorties: tuple[SortieSummary, ...]
    coverage: tuple[ZoneCoverage, ...]


def load_phase3_config(path: str | Path) -> Phase3Config:
    """Load a strict Phase 3 YAML configuration."""

    source_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DroneConfigError(f"unable to read {source_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise DroneConfigError(f"invalid YAML in {source_path}: {exc}") from exc

    root = _mapping(raw, "configuration")
    _exact_keys(root, {"phase1_config", "survey", "aircraft", "cameras", "output"}, "configuration")
    phase1_path = _resolve_path(source_path.parent, root["phase1_config"])
    phase1 = load_phase1_config(phase1_path)

    survey = _mapping(root["survey"], "survey")
    _exact_keys(survey, {"date", "start_time", "grounded"}, "survey")
    aircraft = _mapping(root["aircraft"], "aircraft")
    _exact_keys(
        aircraft,
        {
            "name",
            "reference_mass_g",
            "maximum_flight_time_minutes",
            "maximum_speed_m_s",
            "mapping_speed_m_s",
            "reserve_fraction",
            "payload_endurance_factor",
            "altitude_m",
            "forward_overlap",
            "side_overlap",
            "turn_overhead_seconds",
            "takeoff_landing_overhead_seconds",
            "battery_change_seconds",
        },
        "aircraft",
    )
    camera_rows = root["cameras"]
    if not isinstance(camera_rows, list):
        raise DroneConfigError("cameras must be a list")
    output_raw = _mapping(root["output"], "output")
    _exact_keys(
        output_raw,
        {
            "directory",
            "flight_lines_filename",
            "sortie_summary_filename",
            "zone_coverage_filename",
        },
        "output",
    )

    try:
        cameras = tuple(_camera_from_mapping(row, index) for index, row in enumerate(camera_rows, 1))
        config = Phase3Config(
            source_path=source_path,
            phase1=phase1,
            survey_date=_date_value(survey["date"], "survey.date"),
            start_time=_time_value(survey["start_time"], "survey.start_time"),
            grounded=_boolean(survey["grounded"], "survey.grounded"),
            drone=DroneConfig(
                aircraft_name=_text(aircraft["name"], "aircraft.name"),
                reference_mass_g=_number(aircraft["reference_mass_g"], "aircraft.reference_mass_g"),
                maximum_flight_time_minutes=_number(aircraft["maximum_flight_time_minutes"], "aircraft.maximum_flight_time_minutes"),
                maximum_speed_m_s=_number(aircraft["maximum_speed_m_s"], "aircraft.maximum_speed_m_s"),
                mapping_speed_m_s=_number(aircraft["mapping_speed_m_s"], "aircraft.mapping_speed_m_s"),
                reserve_fraction=_number(aircraft["reserve_fraction"], "aircraft.reserve_fraction"),
                payload_endurance_factor=_number(aircraft["payload_endurance_factor"], "aircraft.payload_endurance_factor"),
                altitude_m=_number(aircraft["altitude_m"], "aircraft.altitude_m"),
                forward_overlap=_number(aircraft["forward_overlap"], "aircraft.forward_overlap"),
                side_overlap=_number(aircraft["side_overlap"], "aircraft.side_overlap"),
                turn_overhead_seconds=_number(aircraft["turn_overhead_seconds"], "aircraft.turn_overhead_seconds"),
                takeoff_landing_overhead_seconds=_number(
                    aircraft["takeoff_landing_overhead_seconds"],
                    "aircraft.takeoff_landing_overhead_seconds",
                ),
                battery_change_seconds=_number(aircraft["battery_change_seconds"], "aircraft.battery_change_seconds"),
            ),
            cameras=cameras,
            output=Phase3OutputConfig(
                directory=_resolve_path(source_path.parent, output_raw["directory"]),
                flight_lines_filename=_text(output_raw["flight_lines_filename"], "output.flight_lines_filename"),
                sortie_summary_filename=_text(output_raw["sortie_summary_filename"], "output.sortie_summary_filename"),
                zone_coverage_filename=_text(output_raw["zone_coverage_filename"], "output.zone_coverage_filename"),
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, DroneConfigError):
            raise
        raise DroneConfigError(str(exc)) from exc
    return config


def plan_mission(config: Phase3Config, *, farm: Farm | None = None) -> MissionPlan:
    """Create flight lines, sorties, and coverage for one survey."""

    farm = farm or load_farm(config.phase1)
    drone = config.drone
    footprint_width = min(
        camera.ground_width_m(drone.altitude_m) for camera in config.cameras
    )
    footprint_length = min(
        camera.ground_length_m(drone.altitude_m) for camera in config.cameras
    )
    maximum_line_spacing = footprint_width * (1.0 - drone.side_overlap)
    capture_spacing = footprint_length * (1.0 - drone.forward_overlap)
    capture_interval = capture_spacing / drone.mapping_speed_m_s
    slowest_camera = max(
        camera.minimum_capture_interval_seconds for camera in config.cameras
    )
    if capture_interval < slowest_camera:
        raise ValueError(
            "mapping speed and forward overlap require captures faster than "
            "the configured cameras support"
        )

    if config.grounded:
        coverage = tuple(
            ZoneCoverage(zone.zone_id, False, 0.0, (), None, None)
            for zone in sorted(farm.zones, key=lambda item: (item.row, item.column))
        )
        return MissionPlan(
            survey_date=config.survey_date,
            grounded=True,
            footprint_width_m=footprint_width,
            footprint_length_m=footprint_length,
            line_spacing_m=maximum_line_spacing,
            capture_spacing_m=capture_spacing,
            capture_interval_seconds=capture_interval,
            usable_endurance_seconds=drone.usable_endurance_seconds,
            lines=(),
            sorties=(),
            coverage=coverage,
        )

    line_positions = _line_positions(
        field_height_m=farm.height_m,
        footprint_width_m=footprint_width,
        maximum_spacing_m=maximum_line_spacing,
    )
    actual_line_spacing = (
        line_positions[1] - line_positions[0]
        if len(line_positions) > 1
        else 0.0
    )
    captures_per_line = ceil(farm.width_m / capture_spacing) + 1
    lines, sorties = _assign_sorties(
        line_positions=line_positions,
        field_width_m=farm.width_m,
        captures_per_line=captures_per_line,
        camera_count=len(config.cameras),
        drone=drone,
    )
    coverage = _zone_coverage(
        farm=farm,
        lines=lines,
        footprint_width_m=footprint_width,
        mapping_speed_m_s=drone.mapping_speed_m_s,
        survey_start=datetime.combine(config.survey_date, config.start_time),
    )
    return MissionPlan(
        survey_date=config.survey_date,
        grounded=False,
        footprint_width_m=footprint_width,
        footprint_length_m=footprint_length,
        line_spacing_m=actual_line_spacing,
        capture_spacing_m=capture_spacing,
        capture_interval_seconds=capture_interval,
        usable_endurance_seconds=drone.usable_endurance_seconds,
        lines=lines,
        sorties=sorties,
        coverage=coverage,
    )


def _line_positions(
    *, field_height_m: float, footprint_width_m: float, maximum_spacing_m: float
) -> tuple[float, ...]:
    if footprint_width_m >= field_height_m:
        return (field_height_m / 2.0,)
    span = field_height_m - footprint_width_m
    intervals = ceil(span / maximum_spacing_m)
    spacing = span / intervals
    first = footprint_width_m / 2.0
    return tuple(first + index * spacing for index in range(intervals + 1))


def _assign_sorties(
    *,
    line_positions: tuple[float, ...],
    field_width_m: float,
    captures_per_line: int,
    camera_count: int,
    drone: DroneConfig,
) -> tuple[tuple[FlightLine, ...], tuple[SortieSummary, ...]]:
    lines: list[FlightLine] = []
    summaries: list[SortieSummary] = []
    sortie_lines: list[FlightLine] = []
    sortie_elapsed = drone.takeoff_landing_overhead_seconds
    sortie_distance = 0.0
    mission_clock = 0.0
    previous_y: float | None = None
    sortie_number = 1

    def finish_sortie() -> None:
        nonlocal mission_clock, sortie_lines, sortie_elapsed, sortie_distance
        if not sortie_lines:
            return
        capture_count = sum(line.capture_count_per_camera for line in sortie_lines)
        summaries.append(
            SortieSummary(
                sortie_id=f"S{sortie_number:02d}",
                line_count=len(sortie_lines),
                route_distance_m=sortie_distance,
                flight_time_seconds=sortie_elapsed,
                usable_endurance_seconds=drone.usable_endurance_seconds,
                battery_margin_seconds=drone.usable_endurance_seconds - sortie_elapsed,
                capture_count_per_camera=capture_count,
                total_image_count=capture_count * camera_count,
            )
        )
        mission_clock += sortie_elapsed + drone.battery_change_seconds
        sortie_lines = []
        sortie_elapsed = drone.takeoff_landing_overhead_seconds
        sortie_distance = 0.0

    line_time = field_width_m / drone.mapping_speed_m_s
    for index, y_m in enumerate(line_positions, 1):
        transition_distance = 0.0 if not sortie_lines else abs(y_m - previous_y)
        transition_time = (
            0.0
            if not sortie_lines
            else transition_distance / drone.mapping_speed_m_s
            + drone.turn_overhead_seconds
        )
        addition = transition_time + line_time
        if sortie_lines and sortie_elapsed + addition > drone.usable_endurance_seconds:
            finish_sortie()
            sortie_number += 1
            previous_y = None
            transition_distance = 0.0
            transition_time = 0.0
            addition = line_time
        if sortie_elapsed + addition > drone.usable_endurance_seconds:
            raise ValueError("one flight line does not fit within usable endurance")

        line_start = mission_clock + sortie_elapsed + transition_time
        start_x, end_x = (0.0, field_width_m) if index % 2 else (field_width_m, 0.0)
        line = FlightLine(
            line_id=f"L{index:03d}",
            sortie_id=f"S{sortie_number:02d}",
            sequence=index,
            y_m=y_m,
            start_x_m=start_x,
            end_x_m=end_x,
            distance_m=field_width_m,
            direction="east" if index % 2 else "west",
            capture_count_per_camera=captures_per_line,
            start_seconds=line_start,
            end_seconds=line_start + line_time,
        )
        lines.append(line)
        sortie_lines.append(line)
        sortie_elapsed += addition
        sortie_distance += transition_distance + field_width_m
        previous_y = y_m

    finish_sortie()
    return tuple(lines), tuple(summaries)


def _zone_coverage(
    *,
    farm: Farm,
    lines: tuple[FlightLine, ...],
    footprint_width_m: float,
    mapping_speed_m_s: float,
    survey_start: datetime,
) -> tuple[ZoneCoverage, ...]:
    records: list[ZoneCoverage] = []
    half_width = footprint_width_m / 2.0
    for zone in sorted(farm.zones, key=lambda item: (item.row, item.column)):
        intersecting: list[FlightLine] = []
        intervals: list[tuple[float, float]] = []
        for line in lines:
            lower = max(zone.y_m, line.y_m - half_width)
            upper = min(zone.y_max_m, line.y_m + half_width)
            if upper > lower:
                intersecting.append(line)
                intervals.append((lower, upper))
        covered_height = _merged_length(intervals)
        fraction = min(1.0, covered_height / zone.height_m)
        if intersecting:
            center_x = zone.x_m + zone.width_m / 2.0
            observation_seconds = min(
                line.start_seconds
                + abs(center_x - line.start_x_m) / mapping_speed_m_s
                for line in intersecting
            )
            observation_time = survey_start + timedelta(seconds=observation_seconds)
        else:
            observation_seconds = None
            observation_time = None
        records.append(
            ZoneCoverage(
                zone_id=zone.zone_id,
                covered=fraction > 0.0,
                coverage_fraction=fraction,
                line_ids=tuple(line.line_id for line in intersecting),
                first_observation_seconds=observation_seconds,
                first_observation_time=observation_time,
            )
        )
    return tuple(records)


def _merged_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _camera_from_mapping(value: Any, index: int) -> CameraConfig:
    row = _mapping(value, f"cameras[{index}]")
    expected = {
        "name",
        "channel",
        "width_px",
        "height_px",
        "horizontal_fov_deg",
        "vertical_fov_deg",
        "minimum_capture_interval_seconds",
    }
    _exact_keys(row, expected, f"cameras[{index}]")
    return CameraConfig(
        name=_text(row["name"], f"cameras[{index}].name"),
        channel=_text(row["channel"], f"cameras[{index}].channel"),
        width_px=_integer(row["width_px"], f"cameras[{index}].width_px"),
        height_px=_integer(row["height_px"], f"cameras[{index}].height_px"),
        horizontal_fov_deg=_number(row["horizontal_fov_deg"], f"cameras[{index}].horizontal_fov_deg"),
        vertical_fov_deg=_number(row["vertical_fov_deg"], f"cameras[{index}].vertical_fov_deg"),
        minimum_capture_interval_seconds=_number(
            row["minimum_capture_interval_seconds"],
            f"cameras[{index}].minimum_capture_interval_seconds",
        ),
    )


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DroneConfigError(f"{context} must be a mapping")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = expected - set(mapping)
    extra = set(mapping) - expected
    if missing or extra:
        raise DroneConfigError(
            f"{context} keys do not match schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _resolve_path(base: Path, value: Any) -> Path:
    text = _text(value, "configured path")
    path = Path(text)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DroneConfigError(f"{name} must be a nonempty string")
    return value.strip()


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DroneConfigError(f"{name} must be numeric")
    return float(value)


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DroneConfigError(f"{name} must be an integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DroneConfigError(f"{name} must be boolean")
    return value


def _date_value(value: Any, name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise DroneConfigError(f"{name} must use YYYY-MM-DD") from exc
    raise DroneConfigError(f"{name} must be a date")


def _time_value(value: Any, name: str) -> time:
    if not isinstance(value, str):
        raise DroneConfigError(f"{name} must use HH:MM:SS")
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise DroneConfigError(f"{name} must use HH:MM:SS") from exc


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_number(value: float, name: str) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _nonnegative_number(value: float, name: str) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _fraction_below_one(value: float, name: str) -> None:
    if not isfinite(value) or not 0 <= value < 1:
        raise ValueError(f"{name} must be at least 0 and less than 1")
