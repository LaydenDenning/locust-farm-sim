"""Tests for the common analysis grid."""

from pathlib import Path
import unittest

from src.farm import load_farm, load_phase1_config
from src.simulation.spatial import build_analysis_grid, contiguous_cells, zone_fractions


ROOT = Path(__file__).resolve().parents[1]


class SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.farm = load_farm(load_phase1_config(ROOT / "config" / "phase1.yaml"))
        cls.cells = build_analysis_grid(cls.farm, 10)

    def test_grid_conserves_area_and_assigns_zones(self) -> None:
        self.assertEqual(len(self.cells), 6400)
        self.assertAlmostEqual(sum(cell.area_m2 for cell in self.cells), self.farm.width_m * self.farm.height_m)
        self.assertEqual({cell.zone_id for cell in self.cells}, {zone.zone_id for zone in self.farm.zones})

    def test_cell_footprint_is_contiguous_and_rolls_up(self) -> None:
        selected = contiguous_cells(self.cells, self.cells[0].cell_id, 10)
        fractions = zone_fractions(self.cells, selected)
        self.assertAlmostEqual(sum(fractions.values()), 10 / 256)

    def test_nondividing_cell_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_analysis_grid(self.farm, 13)


if __name__ == "__main__":
    unittest.main()
