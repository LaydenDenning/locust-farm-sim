"""Lightweight analysis grid shared by monitoring methods.

Crop truth remains at management-zone resolution.  Grid cells inherit their
zone's truth and provide a common spatial support for coverage and issue-area
calculations without attempting to simulate raw imagery pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Iterable, Mapping

from src.farm import Farm


@dataclass(frozen=True)
class AnalysisCell:
    cell_id: str
    row: int
    column: int
    x_m: float
    y_m: float
    size_m: float
    zone_id: str

    @property
    def center_x_m(self) -> float:
        return self.x_m + self.size_m / 2.0

    @property
    def center_y_m(self) -> float:
        return self.y_m + self.size_m / 2.0

    @property
    def area_m2(self) -> float:
        return self.size_m * self.size_m


def build_analysis_grid(farm: Farm, cell_size_m: float = 10.0) -> tuple[AnalysisCell, ...]:
    """Partition a rectangular farm into square cells assigned to zones."""

    if cell_size_m <= 0:
        raise ValueError("analysis cell size must be positive")
    columns = round(farm.width_m / cell_size_m)
    rows = round(farm.height_m / cell_size_m)
    if not isclose(columns * cell_size_m, farm.width_m, abs_tol=1e-9) or not isclose(
        rows * cell_size_m, farm.height_m, abs_tol=1e-9
    ):
        raise ValueError("analysis cell size must divide both field dimensions")

    cells: list[AnalysisCell] = []
    for row in range(rows):
        for column in range(columns):
            x_m = column * cell_size_m
            y_m = row * cell_size_m
            center_x = x_m + cell_size_m / 2.0
            center_y = y_m + cell_size_m / 2.0
            zone = next(
                (
                    item
                    for item in farm.zones
                    if item.x_m <= center_x < item.x_max_m
                    and item.y_m <= center_y < item.y_max_m
                ),
                None,
            )
            if zone is None:
                raise ValueError(f"analysis cell at ({x_m}, {y_m}) has no zone")
            cells.append(
                AnalysisCell(
                    cell_id=f"C_R{row + 1:03d}_C{column + 1:03d}",
                    row=row + 1,
                    column=column + 1,
                    x_m=x_m,
                    y_m=y_m,
                    size_m=cell_size_m,
                    zone_id=zone.zone_id,
                )
            )
    return tuple(cells)


def zone_fractions(
    cells: Iterable[AnalysisCell], selected_cell_ids: Iterable[str]
) -> Mapping[str, float]:
    """Return the selected fraction of each represented management zone."""

    cell_rows = tuple(cells)
    selected = set(selected_cell_ids)
    known = {cell.cell_id for cell in cell_rows}
    unknown = selected - known
    if unknown:
        raise ValueError(f"unknown analysis cells: {sorted(unknown)}")
    totals: dict[str, int] = {}
    hits: dict[str, int] = {}
    for cell in cell_rows:
        totals[cell.zone_id] = totals.get(cell.zone_id, 0) + 1
        if cell.cell_id in selected:
            hits[cell.zone_id] = hits.get(cell.zone_id, 0) + 1
    return {
        zone_id: hits.get(zone_id, 0) / count
        for zone_id, count in sorted(totals.items())
    }


def contiguous_cells(
    cells: tuple[AnalysisCell, ...], start_cell_id: str, count: int
) -> tuple[str, ...]:
    """Select a deterministic row/column-contiguous cell footprint."""

    if count < 1 or count > len(cells):
        raise ValueError("cell footprint count is outside the grid")
    by_id = {cell.cell_id: cell for cell in cells}
    if start_cell_id not in by_id:
        raise ValueError(f"unknown start cell: {start_cell_id}")
    by_position = {(cell.row, cell.column): cell for cell in cells}
    selected = [by_id[start_cell_id]]
    seen = {start_cell_id}
    cursor = 0
    while len(selected) < count:
        cell = selected[cursor]
        cursor += 1
        for position in (
            (cell.row, cell.column + 1),
            (cell.row + 1, cell.column),
            (cell.row, cell.column - 1),
            (cell.row - 1, cell.column),
        ):
            neighbor = by_position.get(position)
            if neighbor is not None and neighbor.cell_id not in seen:
                selected.append(neighbor)
                seen.add(neighbor.cell_id)
                if len(selected) == count:
                    break
    return tuple(cell.cell_id for cell in selected)
