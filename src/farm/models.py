"""Immutable data models for the Phase 1 spatial farm."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite


@dataclass(frozen=True)
class FieldZone:
    """Inputs and rectangular geometry for one simulation zone.

    ``x_m`` and ``y_m`` identify the lower-left corner of the zone within the
    field coordinate system.
    """

    zone_id: str
    row: int
    column: int
    x_m: float
    y_m: float
    width_m: float
    height_m: float
    soil_profile: str
    planting_offset_days: int
    initial_available_n_kg_ha: float
    stand_density_plants_m2: float
    slow_drainage: bool

    def __post_init__(self) -> None:
        if not self.zone_id.strip():
            raise ValueError("zone_id must not be empty")
        if self.row < 1 or self.column < 1:
            raise ValueError(f"{self.zone_id}: row and column must be positive")
        for name in ("x_m", "y_m", "width_m", "height_m"):
            value = getattr(self, name)
            if not isfinite(value):
                raise ValueError(f"{self.zone_id}: {name} must be finite")
        if self.x_m < 0 or self.y_m < 0:
            raise ValueError(f"{self.zone_id}: coordinates must be nonnegative")
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError(f"{self.zone_id}: zone dimensions must be positive")
        if not self.soil_profile.strip():
            raise ValueError(f"{self.zone_id}: soil_profile must not be empty")
        if not isinstance(self.planting_offset_days, int) or isinstance(
            self.planting_offset_days, bool
        ):
            raise ValueError(f"{self.zone_id}: planting_offset_days must be an integer")
        if not isfinite(self.initial_available_n_kg_ha):
            raise ValueError(
                f"{self.zone_id}: initial_available_n_kg_ha must be finite"
            )
        if self.initial_available_n_kg_ha < 0:
            raise ValueError(
                f"{self.zone_id}: initial_available_n_kg_ha must be nonnegative"
            )
        if not isfinite(self.stand_density_plants_m2):
            raise ValueError(f"{self.zone_id}: stand density must be finite")
        if self.stand_density_plants_m2 <= 0:
            raise ValueError(f"{self.zone_id}: stand density must be positive")
        if not isinstance(self.slow_drainage, bool):
            raise ValueError(f"{self.zone_id}: slow_drainage must be boolean")

    @property
    def x_max_m(self) -> float:
        return self.x_m + self.width_m

    @property
    def y_max_m(self) -> float:
        return self.y_m + self.height_m

    @property
    def area_m2(self) -> float:
        return self.width_m * self.height_m

    @property
    def tdwi_kg_ha(self) -> float:
        """Initial dry weight scaled from the approved 8 plants/m2 baseline."""

        return 50.0 * self.stand_density_plants_m2 / 8.0


@dataclass(frozen=True)
class Farm:
    """A rectangular field completely partitioned into simulation zones."""

    width_m: float
    height_m: float
    rows: int
    columns: int
    zones: tuple[FieldZone, ...]

    def __post_init__(self) -> None:
        if not isfinite(self.width_m) or not isfinite(self.height_m):
            raise ValueError("field dimensions must be finite")
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("field dimensions must be positive")
        if (
            not isinstance(self.rows, int)
            or isinstance(self.rows, bool)
            or not isinstance(self.columns, int)
            or isinstance(self.columns, bool)
            or self.rows < 1
            or self.columns < 1
        ):
            raise ValueError("field rows and columns must be positive")

        object.__setattr__(self, "zones", tuple(self.zones))

        expected_count = self.rows * self.columns
        if len(self.zones) != expected_count:
            raise ValueError(
                f"farm requires {expected_count} zones, found {len(self.zones)}"
            )

        zone_ids = [zone.zone_id for zone in self.zones]
        if len(set(zone_ids)) != len(zone_ids):
            raise ValueError("zone_id values must be unique")

        cells = [(zone.row, zone.column) for zone in self.zones]
        if len(set(cells)) != len(cells):
            raise ValueError("zone row/column values must be unique")
        expected_cells = {
            (row, column)
            for row in range(1, self.rows + 1)
            for column in range(1, self.columns + 1)
        }
        actual_cells = set(cells)
        if actual_cells != expected_cells:
            missing = sorted(expected_cells - actual_cells)
            extra = sorted(actual_cells - expected_cells)
            raise ValueError(
                f"zone grid does not match field; missing={missing}, extra={extra}"
            )

        for zone in self.zones:
            if zone.x_max_m > self.width_m or zone.y_max_m > self.height_m:
                raise ValueError(f"{zone.zone_id}: geometry extends outside the field")

        for index, first in enumerate(self.zones):
            for second in self.zones[index + 1 :]:
                if _rectangles_overlap(first, second):
                    raise ValueError(
                        f"zone geometries overlap: {first.zone_id} and {second.zone_id}"
                    )

        field_area = self.width_m * self.height_m
        zone_area = sum(zone.area_m2 for zone in self.zones)
        if not isclose(zone_area, field_area, rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(
                f"zones do not cover the complete field: {zone_area} of {field_area} m2"
            )

    def get_zone(self, zone_id: str) -> FieldZone:
        """Return a zone by ID, raising ``KeyError`` when it is absent."""

        for zone in self.zones:
            if zone.zone_id == zone_id:
                return zone
        raise KeyError(zone_id)


def _rectangles_overlap(first: FieldZone, second: FieldZone) -> bool:
    return not (
        first.x_max_m <= second.x_m
        or second.x_max_m <= first.x_m
        or first.y_max_m <= second.y_m
        or second.y_max_m <= first.y_m
    )
