"""Reusable aerial and ground route generators."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot
from random import Random
from typing import Iterable

from src.farm import Farm


@dataclass(frozen=True)
class AerialLeg:
    sequence: int
    start_x_m: float
    start_y_m: float
    end_x_m: float
    end_y_m: float

    @property
    def distance_m(self) -> float:
        return hypot(self.end_x_m - self.start_x_m, self.end_y_m - self.start_y_m)


def aerial_parallel_sweep(
    farm: Farm,
    *,
    footprint_width_m: float,
    side_overlap: float,
    orientation: str = "east_west",
    target_zone_ids: Iterable[str] = (),
) -> tuple[AerialLeg, ...]:
    """Generate alternating parallel legs over the field or selected zones."""

    if footprint_width_m <= 0:
        raise ValueError("footprint width must be positive")
    if not 0 <= side_overlap < 1:
        raise ValueError("side overlap must be at least zero and below one")
    if orientation not in {"east_west", "north_south"}:
        raise ValueError("orientation must be east_west or north_south")
    target_ids = tuple(target_zone_ids)
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("target zones must be unique")
    known = {zone.zone_id for zone in farm.zones}
    unknown = set(target_ids) - known
    if unknown:
        raise ValueError(f"unknown target zones: {sorted(unknown)}")

    regions = (
        [(0.0, farm.width_m, 0.0, farm.height_m)]
        if not target_ids
        else [
            (zone.x_m, zone.x_max_m, zone.y_m, zone.y_max_m)
            for zone_id in target_ids
            for zone in (farm.get_zone(zone_id),)
        ]
    )
    raw: list[tuple[float, float, float, float]] = []
    spacing = footprint_width_m * (1.0 - side_overlap)
    for x0, x1, y0, y1 in regions:
        cross_start, cross_end = (y0, y1) if orientation == "east_west" else (x0, x1)
        span = cross_end - cross_start
        if footprint_width_m >= span:
            positions = (cross_start + span / 2.0,)
        else:
            intervals = ceil((span - footprint_width_m) / spacing)
            actual = (span - footprint_width_m) / max(1, intervals)
            first = cross_start + footprint_width_m / 2.0
            positions = tuple(first + index * actual for index in range(intervals + 1))
        for position in positions:
            if orientation == "east_west":
                raw.append((x0, position, x1, position))
            else:
                raw.append((position, y0, position, y1))

    unique = list(dict.fromkeys(raw))
    legs: list[AerialLeg] = []
    for index, points in enumerate(unique, start=1):
        x0, y0, x1, y1 = points
        if index % 2 == 0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        legs.append(AerialLeg(index, x0, y0, x1, y1))
    return tuple(legs)


def ground_route(
    farm: Farm,
    *,
    pattern: str,
    zone_ids: Iterable[str] = (),
    sample_count: int | None = None,
    seed: int = 0,
    samples_per_leg: int = 3,
) -> tuple[str, ...]:
    """Build an explicit, W, serpentine, or seeded-sample zone route."""

    ordered = sorted(farm.zones, key=lambda zone: (zone.row, zone.column))
    known = {zone.zone_id for zone in ordered}
    if pattern == "explicit":
        result = tuple(zone_ids)
        if not result:
            raise ValueError("explicit route requires zone_ids")
    elif pattern == "serpentine":
        result = tuple(
            zone.zone_id
            for row in range(1, farm.rows + 1)
            for zone in sorted(
                (item for item in ordered if item.row == row),
                key=lambda item: item.column,
                reverse=row % 2 == 0,
            )
        )
    elif pattern == "seeded_sample":
        if sample_count is None or not 1 <= sample_count <= len(ordered):
            raise ValueError("seeded sample_count must fit within the farm")
        result = tuple(zone.zone_id for zone in Random(seed).sample(ordered, sample_count))
    elif pattern == "w":
        if samples_per_leg < 2:
            raise ValueError("W route samples_per_leg must be at least two")
        vertices = (
            (0.0, farm.height_m),
            (farm.width_m * 0.25, 0.0),
            (farm.width_m * 0.50, farm.height_m),
            (farm.width_m * 0.75, 0.0),
            (farm.width_m, farm.height_m),
        )
        selected: list[str] = []
        for leg_index, (start, end) in enumerate(zip(vertices, vertices[1:])):
            for step in range(samples_per_leg):
                if leg_index and step == 0:
                    continue
                fraction = step / (samples_per_leg - 1)
                x = start[0] + fraction * (end[0] - start[0])
                y = start[1] + fraction * (end[1] - start[1])
                x = min(x, farm.width_m - 1e-9)
                y = min(y, farm.height_m - 1e-9)
                zone = next(
                    item
                    for item in ordered
                    if item.x_m <= x < item.x_max_m and item.y_m <= y < item.y_max_m
                )
                if zone.zone_id not in selected:
                    selected.append(zone.zone_id)
        result = tuple(selected)
    else:
        raise ValueError(f"unknown ground route pattern: {pattern}")

    if len(set(result)) != len(result):
        raise ValueError("ground route zones must be unique")
    unknown = set(result) - known
    if unknown:
        raise ValueError(f"ground route contains unknown zones: {sorted(unknown)}")
    return result
