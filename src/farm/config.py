"""Configuration and CSV loading for the Phase 1 farm."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .models import Farm, FieldZone


class ConfigError(ValueError):
    """Raised when a Phase 1 configuration or zone file is invalid."""


@dataclass(frozen=True)
class FieldConfig:
    width_m: float
    height_m: float
    rows: int
    columns: int

    def __post_init__(self) -> None:
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("field dimensions must be positive")
        if self.rows < 1 or self.columns < 1:
            raise ValueError("field rows and columns must be positive")


@dataclass(frozen=True)
class SoilProfile:
    """Synthetic single-layer soil inputs used by the classic water balance."""

    name: str
    smw: float
    smfcf: float
    sm0: float
    rdmsol_cm: float
    crairc: float
    k0_cm_day: float
    sope_cm_day: float
    ksub_cm_day: float
    base_wav_cm: float
    ssmax_cm: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("soil profile name must not be empty")
        if not 0 < self.smw < self.smfcf < self.sm0 <= 1:
            raise ValueError(
                f"{self.name}: expected 0 < SMW < SMFCF < SM0 <= 1"
            )
        if self.rdmsol_cm <= 0:
            raise ValueError(f"{self.name}: RDMSOL must be positive")
        if not 0 <= self.crairc <= 1:
            raise ValueError(f"{self.name}: CRAIRC must be between 0 and 1")
        for name in (
            "k0_cm_day",
            "sope_cm_day",
            "ksub_cm_day",
            "base_wav_cm",
            "ssmax_cm",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{self.name}: {name} must be finite and nonnegative")


@dataclass(frozen=True)
class SlowDrainageConfig:
    sope_cm_day: float
    ksub_cm_day: float
    ssmax_cm: float
    wav_addition_cm: float

    def __post_init__(self) -> None:
        for name in (
            "sope_cm_day",
            "ksub_cm_day",
            "ssmax_cm",
            "wav_addition_cm",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"slow_drainage.{name} must be finite and nonnegative")


@dataclass(frozen=True)
class CropConfig:
    parameter_directory: Path
    crop_name: str
    variety_name: str
    model_name: str

    def __post_init__(self) -> None:
        for name in ("crop_name", "variety_name", "model_name"):
            if not getattr(self, name).strip():
                raise ValueError(f"crop.{name} must not be empty")


@dataclass(frozen=True)
class WeatherConfig:
    file: Path
    latitude: float
    longitude: float
    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if not -90 <= self.latitude <= 90:
            raise ValueError("weather.latitude must be between -90 and 90")
        if not -180 <= self.longitude <= 180:
            raise ValueError("weather.longitude must be between -180 and 180")
        if self.start_date > self.end_date:
            raise ValueError("weather.start_date must not follow end_date")


@dataclass(frozen=True)
class CalendarConfig:
    base_sowing_date: date
    crop_start_type: str
    crop_end_type: str
    max_duration_days: int

    def __post_init__(self) -> None:
        if self.crop_start_type != "sowing":
            raise ValueError("Phase 1 calendar.crop_start_type must be 'sowing'")
        if self.crop_end_type != "maturity":
            raise ValueError("Phase 1 calendar.crop_end_type must be 'maturity'")
        if self.max_duration_days < 1:
            raise ValueError("calendar.max_duration_days must be positive")


@dataclass(frozen=True)
class SiteConfig:
    co2_ppm: float

    def __post_init__(self) -> None:
        if not isfinite(self.co2_ppm) or self.co2_ppm <= 0:
            raise ValueError("site.co2_ppm must be finite and positive")


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    daily_truth_filename: str
    zone_summary_filename: str
    lai_plot_filename: str
    soil_moisture_plot_filename: str
    nitrogen_plot_filename: str
    yield_heatmap_filename: str

    def __post_init__(self) -> None:
        for name in (
            "daily_truth_filename",
            "zone_summary_filename",
            "lai_plot_filename",
            "soil_moisture_plot_filename",
            "nitrogen_plot_filename",
            "yield_heatmap_filename",
        ):
            value = getattr(self, name)
            if not value.strip() or Path(value).name != value:
                raise ValueError(f"output.{name} must be a simple file name")


@dataclass(frozen=True)
class Phase1Config:
    source_path: Path
    project_root: Path
    zones_file: Path
    field: FieldConfig
    crop: CropConfig
    weather: WeatherConfig
    calendar: CalendarConfig
    site: SiteConfig
    soil_profiles: Mapping[str, SoilProfile]
    slow_drainage: SlowDrainageConfig
    output: OutputConfig

    def __post_init__(self) -> None:
        if not self.soil_profiles:
            raise ValueError("at least one soil profile is required")
        object.__setattr__(
            self, "soil_profiles", MappingProxyType(dict(self.soil_profiles))
        )


ZONE_COLUMNS = (
    "zone_id",
    "row",
    "column",
    "x_m",
    "y_m",
    "width_m",
    "height_m",
    "soil_profile",
    "planting_offset_days",
    "initial_available_n_kg_ha",
    "stand_density_plants_m2",
    "slow_drainage",
)


def load_phase1_config(path: str | Path) -> Phase1Config:
    """Load and validate the Phase 1 YAML configuration."""

    source_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"unable to read configuration {source_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source_path}: {exc}") from exc

    root = _mapping(raw, "configuration")
    project_root_value = root.get("project_root", ".")
    project_root = _resolve_path(source_path.parent, project_root_value, "project_root")

    try:
        field_raw = _mapping(_required(root, "field"), "field")
        crop_raw = _mapping(_required(root, "crop"), "crop")
        weather_raw = _mapping(_required(root, "weather"), "weather")
        calendar_raw = _mapping(_required(root, "calendar"), "calendar")
        site_raw = _mapping(_required(root, "site"), "site")
        soils_raw = _mapping(_required(root, "soil_profiles"), "soil_profiles")
        drainage_raw = _mapping(
            _required(root, "slow_drainage"), "slow_drainage"
        )
        output_raw = _mapping(_required(root, "output"), "output")

        field = FieldConfig(
            width_m=_float(field_raw, "width_m", "field"),
            height_m=_float(field_raw, "height_m", "field"),
            rows=_int(field_raw, "rows", "field"),
            columns=_int(field_raw, "columns", "field"),
        )
        crop = CropConfig(
            parameter_directory=_resolve_path(
                project_root,
                _required(crop_raw, "parameter_directory"),
                "crop.parameter_directory",
            ),
            crop_name=_string(crop_raw, "crop_name", "crop"),
            variety_name=_string(crop_raw, "variety_name", "crop"),
            model_name=_string(crop_raw, "model_name", "crop"),
        )
        weather = WeatherConfig(
            file=_resolve_path(
                project_root,
                _required(weather_raw, "file"),
                "weather.file",
            ),
            latitude=_float(weather_raw, "latitude", "weather"),
            longitude=_float(weather_raw, "longitude", "weather"),
            start_date=_date(weather_raw, "start_date", "weather"),
            end_date=_date(weather_raw, "end_date", "weather"),
        )
        calendar = CalendarConfig(
            base_sowing_date=_date(
                calendar_raw, "base_sowing_date", "calendar"
            ),
            crop_start_type=_string(
                calendar_raw, "crop_start_type", "calendar"
            ),
            crop_end_type=_string(calendar_raw, "crop_end_type", "calendar"),
            max_duration_days=_int(
                calendar_raw, "max_duration_days", "calendar"
            ),
        )
        site = SiteConfig(co2_ppm=_float(site_raw, "co2_ppm", "site"))
        soil_profiles = {
            str(name): _soil_profile(str(name), _mapping(value, f"soil_profiles.{name}"))
            for name, value in soils_raw.items()
        }
        slow_drainage = SlowDrainageConfig(
            sope_cm_day=_float(drainage_raw, "sope_cm_day", "slow_drainage"),
            ksub_cm_day=_float(drainage_raw, "ksub_cm_day", "slow_drainage"),
            ssmax_cm=_float(drainage_raw, "ssmax_cm", "slow_drainage"),
            wav_addition_cm=_float(
                drainage_raw, "wav_addition_cm", "slow_drainage"
            ),
        )
        output = OutputConfig(
            directory=_resolve_path(
                project_root,
                _required(output_raw, "directory"),
                "output.directory",
            ),
            daily_truth_filename=_string(
                output_raw, "daily_truth_filename", "output"
            ),
            zone_summary_filename=_string(
                output_raw, "zone_summary_filename", "output"
            ),
            lai_plot_filename=_string(
                output_raw, "lai_plot_filename", "output"
            ),
            soil_moisture_plot_filename=_string(
                output_raw, "soil_moisture_plot_filename", "output"
            ),
            nitrogen_plot_filename=_string(
                output_raw, "nitrogen_plot_filename", "output"
            ),
            yield_heatmap_filename=_string(
                output_raw, "yield_heatmap_filename", "output"
            ),
        )

        return Phase1Config(
            source_path=source_path,
            project_root=project_root,
            zones_file=_resolve_path(
                project_root, _required(root, "zones_file"), "zones_file"
            ),
            field=field,
            crop=crop,
            weather=weather,
            calendar=calendar,
            site=site,
            soil_profiles=soil_profiles,
            slow_drainage=slow_drainage,
            output=output,
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(str(exc)) from exc


def load_zones_csv(path: str | Path) -> tuple[FieldZone, ...]:
    """Load zones from a CSV with a deliberately strict, inspectable schema."""

    source_path = Path(path)
    try:
        with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ConfigError(f"zone CSV {source_path} has no header")
            missing = set(ZONE_COLUMNS) - set(reader.fieldnames)
            extra = set(reader.fieldnames) - set(ZONE_COLUMNS)
            if missing or extra:
                raise ConfigError(
                    f"zone CSV columns do not match schema; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
            zones = tuple(
                _zone_from_row(row, line_number)
                for line_number, row in enumerate(reader, start=2)
            )
    except OSError as exc:
        raise ConfigError(f"unable to read zone CSV {source_path}: {exc}") from exc

    if not zones:
        raise ConfigError(f"zone CSV {source_path} contains no zones")
    return zones


def load_farm(config: Phase1Config) -> Farm:
    """Load the configured zones and validate cross-file Phase 1 invariants."""

    zones = load_zones_csv(config.zones_file)
    unknown_profiles = sorted(
        {zone.soil_profile for zone in zones} - set(config.soil_profiles)
    )
    if unknown_profiles:
        raise ConfigError(f"zones reference unknown soil profiles: {unknown_profiles}")

    first_sowing = config.calendar.base_sowing_date + timedelta(
        days=min(zone.planting_offset_days for zone in zones)
    )
    last_possible_day = (
        config.calendar.base_sowing_date
        + timedelta(days=max(zone.planting_offset_days for zone in zones))
        + timedelta(days=config.calendar.max_duration_days)
    )
    if config.weather.start_date > first_sowing:
        raise ConfigError("weather coverage starts after the earliest sowing date")
    if config.weather.end_date < last_possible_day:
        raise ConfigError("weather coverage ends before the latest possible crop day")

    try:
        return Farm(
            width_m=config.field.width_m,
            height_m=config.field.height_m,
            rows=config.field.rows,
            columns=config.field.columns,
            zones=zones,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _zone_from_row(row: Mapping[str, str], line_number: int) -> FieldZone:
    context = f"zone CSV line {line_number}"
    try:
        return FieldZone(
            zone_id=_row_string(row, "zone_id", context),
            row=_row_int(row, "row", context),
            column=_row_int(row, "column", context),
            x_m=_row_float(row, "x_m", context),
            y_m=_row_float(row, "y_m", context),
            width_m=_row_float(row, "width_m", context),
            height_m=_row_float(row, "height_m", context),
            soil_profile=_row_string(row, "soil_profile", context),
            planting_offset_days=_row_int(row, "planting_offset_days", context),
            initial_available_n_kg_ha=_row_float(
                row, "initial_available_n_kg_ha", context
            ),
            stand_density_plants_m2=_row_float(
                row, "stand_density_plants_m2", context
            ),
            slow_drainage=_row_bool(row, "slow_drainage", context),
        )
    except ValueError as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"{context}: {exc}") from exc


def _soil_profile(name: str, raw: Mapping[str, Any]) -> SoilProfile:
    return SoilProfile(
        name=name,
        smw=_float(raw, "smw", f"soil_profiles.{name}"),
        smfcf=_float(raw, "smfcf", f"soil_profiles.{name}"),
        sm0=_float(raw, "sm0", f"soil_profiles.{name}"),
        rdmsol_cm=_float(raw, "rdmsol_cm", f"soil_profiles.{name}"),
        crairc=_float(raw, "crairc", f"soil_profiles.{name}"),
        k0_cm_day=_float(raw, "k0_cm_day", f"soil_profiles.{name}"),
        sope_cm_day=_float(raw, "sope_cm_day", f"soil_profiles.{name}"),
        ksub_cm_day=_float(raw, "ksub_cm_day", f"soil_profiles.{name}"),
        base_wav_cm=_float(raw, "base_wav_cm", f"soil_profiles.{name}"),
        ssmax_cm=_float(raw, "ssmax_cm", f"soil_profiles.{name}", default=0.0),
    )


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required configuration key: {key}")
    return mapping[key]


def _string(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _required(mapping, key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context}.{key} must be a nonempty string")
    return value.strip()


def _float(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
    *,
    default: float | None = None,
) -> float:
    if key not in mapping and default is not None:
        return default
    value = _required(mapping, key)
    if isinstance(value, bool):
        raise ConfigError(f"{context}.{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context}.{key} must be numeric") from exc
    if not isfinite(result):
        raise ConfigError(f"{context}.{key} must be finite")
    return result


def _int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key)
    if isinstance(value, bool):
        raise ConfigError(f"{context}.{key} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise ConfigError(f"{context}.{key} must be an integer")


def _date(mapping: Mapping[str, Any], key: str, context: str) -> date:
    value = _required(mapping, key)
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ConfigError(f"{context}.{key} must be an ISO date") from exc
    raise ConfigError(f"{context}.{key} must be an ISO date")


def _resolve_path(base: Path, value: Any, context: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ConfigError(f"{context} must be a nonempty path")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _row_string(row: Mapping[str, str], key: str, context: str) -> str:
    value = row.get(key)
    if value is None or not value.strip():
        raise ConfigError(f"{context}: {key} must not be empty")
    return value.strip()


def _row_float(row: Mapping[str, str], key: str, context: str) -> float:
    value = _row_string(row, key, context)
    try:
        result = float(value)
    except ValueError as exc:
        raise ConfigError(f"{context}: {key} must be numeric") from exc
    if not isfinite(result):
        raise ConfigError(f"{context}: {key} must be finite")
    return result


def _row_int(row: Mapping[str, str], key: str, context: str) -> int:
    value = _row_string(row, key, context)
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{context}: {key} must be an integer") from exc


def _row_bool(row: Mapping[str, str], key: str, context: str) -> bool:
    value = _row_string(row, key, context).lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ConfigError(f"{context}: {key} must be true or false")
