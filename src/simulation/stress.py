"""Controlled stress-event inputs for the Phase 2 truth simulation."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.farm import Farm, Phase1Config, load_farm, load_phase1_config


STRESS_TYPES = ("water_deficit", "nitrogen_deficit", "stand_loss")
INITIAL_STRESS_TYPES = ("nitrogen_deficit", "stand_loss")
STRESS_EVENT_COLUMNS = (
    "event_id",
    "stress_type",
    "zone_id",
    "start_day",
    "duration_days",
    "severity",
)


class StressConfigError(ValueError):
    """Raised when Phase 2 configuration or event inputs are invalid."""


@dataclass(frozen=True)
class StressEvent:
    """One controlled stress treatment applied to one field zone.

    ``start_day`` is zero-based and relative to that zone's planting date.
    Severity is a fraction between zero and one. For a water deficit it is the
    fraction of the field-capacity-to-wilting-point range removed. For the two
    establishment-time treatments it is the fraction of initial N or stand
    biomass removed.
    """

    event_id: str
    stress_type: str
    zone_id: str
    start_day: int
    duration_days: int
    severity: float

    def __post_init__(self) -> None:
        if not self.event_id.strip() or "|" in self.event_id:
            raise ValueError("event_id must be nonempty and must not contain '|'")
        if self.stress_type not in STRESS_TYPES:
            raise ValueError(
                f"{self.event_id}: stress_type must be one of {STRESS_TYPES}"
            )
        if not self.zone_id.strip():
            raise ValueError(f"{self.event_id}: zone_id must not be empty")
        if (
            not isinstance(self.start_day, int)
            or isinstance(self.start_day, bool)
            or self.start_day < 0
        ):
            raise ValueError(
                f"{self.event_id}: start_day must be a nonnegative integer"
            )
        if (
            not isinstance(self.duration_days, int)
            or isinstance(self.duration_days, bool)
            or self.duration_days < 1
        ):
            raise ValueError(
                f"{self.event_id}: duration_days must be a positive integer"
            )
        if not isfinite(self.severity) or not 0 < self.severity < 1:
            raise ValueError(
                f"{self.event_id}: severity must be greater than 0 and less than 1"
            )

    @property
    def end_day(self) -> int:
        """Return the inclusive final day after planting."""

        return self.start_day + self.duration_days - 1

    def start_date(self, planting_date: date) -> date:
        return planting_date + timedelta(days=self.start_day)

    def end_date(self, planting_date: date) -> date:
        return planting_date + timedelta(days=self.end_day)

    def is_active(self, day: date, planting_date: date) -> bool:
        return self.start_date(planting_date) <= day <= self.end_date(planting_date)


@dataclass(frozen=True)
class Phase2OutputConfig:
    directory: Path
    daily_truth_filename: str
    zone_summary_filename: str
    stress_impacts_filename: str
    trajectory_plot_filename: str
    yield_impact_heatmap_filename: str

    def __post_init__(self) -> None:
        for name in (
            "daily_truth_filename",
            "zone_summary_filename",
            "stress_impacts_filename",
            "trajectory_plot_filename",
            "yield_impact_heatmap_filename",
        ):
            value = getattr(self, name)
            if not value.strip() or Path(value).name != value:
                raise ValueError(f"output.{name} must be a simple file name")


@dataclass(frozen=True)
class Phase2Config:
    source_path: Path
    phase1: Phase1Config
    events_file: Path
    events: tuple[StressEvent, ...]
    output: Phase2OutputConfig


def load_phase2_config(path: str | Path) -> Phase2Config:
    """Load Phase 2 settings and validate their referenced Phase 1 farm."""

    source_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise StressConfigError(
            f"unable to read configuration {source_path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise StressConfigError(f"invalid YAML in {source_path}: {exc}") from exc

    root = _mapping(raw, "configuration")
    expected = {"phase1_config", "stress_events_file", "output"}
    missing = expected - set(root)
    extra = set(root) - expected
    if missing or extra:
        raise StressConfigError(
            f"Phase 2 config keys do not match schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    phase1_path = _resolve_path(source_path.parent, root["phase1_config"])
    events_file = _resolve_path(source_path.parent, root["stress_events_file"])
    phase1 = load_phase1_config(phase1_path)
    farm = load_farm(phase1)
    events = load_stress_events(
        events_file,
        farm=farm,
        max_duration_days=phase1.calendar.max_duration_days,
    )

    output_raw = _mapping(root["output"], "output")
    output_keys = {
        "directory",
        "daily_truth_filename",
        "zone_summary_filename",
        "stress_impacts_filename",
        "trajectory_plot_filename",
        "yield_impact_heatmap_filename",
    }
    missing_output = output_keys - set(output_raw)
    extra_output = set(output_raw) - output_keys
    if missing_output or extra_output:
        raise StressConfigError(
            f"Phase 2 output keys do not match schema; "
            f"missing={sorted(missing_output)}, extra={sorted(extra_output)}"
        )
    try:
        output = Phase2OutputConfig(
            directory=_resolve_path(source_path.parent, output_raw["directory"]),
            daily_truth_filename=_text(output_raw, "daily_truth_filename"),
            zone_summary_filename=_text(output_raw, "zone_summary_filename"),
            stress_impacts_filename=_text(output_raw, "stress_impacts_filename"),
            trajectory_plot_filename=_text(output_raw, "trajectory_plot_filename"),
            yield_impact_heatmap_filename=_text(
                output_raw, "yield_impact_heatmap_filename"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise StressConfigError(str(exc)) from exc

    return Phase2Config(
        source_path=source_path,
        phase1=phase1,
        events_file=events_file,
        events=events,
        output=output,
    )


def load_stress_events(
    path: str | Path, *, farm: Farm, max_duration_days: int
) -> tuple[StressEvent, ...]:
    """Load a strict one-row-per-event-zone CSV and validate attribution."""

    source_path = Path(path)
    try:
        with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise StressConfigError(f"stress CSV {source_path} has no header")
            missing = set(STRESS_EVENT_COLUMNS) - set(reader.fieldnames)
            extra = set(reader.fieldnames) - set(STRESS_EVENT_COLUMNS)
            if missing or extra:
                raise StressConfigError(
                    f"stress CSV columns do not match schema; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
            events = tuple(
                _event_from_row(row, line_number)
                for line_number, row in enumerate(reader, start=2)
            )
    except OSError as exc:
        raise StressConfigError(f"unable to read stress CSV {source_path}: {exc}") from exc

    if not events:
        raise StressConfigError(f"stress CSV {source_path} contains no events")
    _validate_events(events, farm=farm, max_duration_days=max_duration_days)
    return events


def events_by_zone(
    events: Sequence[StressEvent],
) -> dict[str, tuple[StressEvent, ...]]:
    grouped: dict[str, list[StressEvent]] = defaultdict(list)
    for event in events:
        grouped[event.zone_id].append(event)
    return {
        zone_id: tuple(sorted(items, key=lambda item: (item.start_day, item.event_id)))
        for zone_id, items in grouped.items()
    }


def _validate_events(
    events: Sequence[StressEvent], *, farm: Farm, max_duration_days: int
) -> None:
    zone_ids = {zone.zone_id for zone in farm.zones}
    keys: set[tuple[str, str]] = set()
    zone_assignments: set[str] = set()
    definitions: dict[str, tuple[object, ...]] = {}

    for event in events:
        if event.zone_id not in zone_ids:
            raise StressConfigError(
                f"{event.event_id}: unknown zone_id {event.zone_id!r}"
            )
        key = (event.event_id, event.zone_id)
        if key in keys:
            raise StressConfigError(
                f"duplicate stress event/zone row: {event.event_id}, {event.zone_id}"
            )
        keys.add(key)
        if event.zone_id in zone_assignments:
            raise StressConfigError(
                f"{event.zone_id} has more than one event; Phase 2 requires "
                "non-overlapping treatments for causal attribution"
            )
        zone_assignments.add(event.zone_id)

        definition = (
            event.stress_type,
            event.start_day,
            event.duration_days,
            event.severity,
        )
        if event.event_id in definitions and definitions[event.event_id] != definition:
            raise StressConfigError(
                f"{event.event_id}: footprint rows must use identical event settings"
            )
        definitions[event.event_id] = definition

        if event.end_day >= max_duration_days:
            raise StressConfigError(
                f"{event.event_id}: event extends beyond the "
                f"{max_duration_days}-day crop campaign"
            )
        if event.stress_type in INITIAL_STRESS_TYPES and (
            event.start_day != 0 or event.duration_days != max_duration_days
        ):
            raise StressConfigError(
                f"{event.event_id}: {event.stress_type} must start on day 0 and "
                f"last all {max_duration_days} campaign days"
            )


def _event_from_row(row: Mapping[str, str], line_number: int) -> StressEvent:
    context = f"stress CSV line {line_number}"
    try:
        return StressEvent(
            event_id=_row_text(row, "event_id", context),
            stress_type=_row_text(row, "stress_type", context),
            zone_id=_row_text(row, "zone_id", context),
            start_day=_row_int(row, "start_day", context),
            duration_days=_row_int(row, "duration_days", context),
            severity=_row_float(row, "severity", context),
        )
    except ValueError as exc:
        if isinstance(exc, StressConfigError):
            raise
        raise StressConfigError(f"{context}: {exc}") from exc


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StressConfigError(f"{context} must be a mapping")
    return value


def _resolve_path(base: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise StressConfigError("configured paths must be nonempty strings")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise StressConfigError(f"output.{key} must be a nonempty string")
    return value.strip()


def _row_text(row: Mapping[str, str], key: str, context: str) -> str:
    value = row.get(key)
    if value is None or not value.strip():
        raise StressConfigError(f"{context}: {key} must not be empty")
    return value.strip()


def _row_int(row: Mapping[str, str], key: str, context: str) -> int:
    value = _row_text(row, key, context)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise StressConfigError(f"{context}: {key} must be an integer") from exc
    return parsed


def _row_float(row: Mapping[str, str], key: str, context: str) -> float:
    value = _row_text(row, key, context)
    try:
        return float(value)
    except ValueError as exc:
        raise StressConfigError(f"{context}: {key} must be numeric") from exc
