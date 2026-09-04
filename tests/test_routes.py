"""Tests for modular route generators."""

from pathlib import Path
import unittest

from src.farm import load_farm, load_phase1_config
from src.simulation.routes import aerial_parallel_sweep, ground_route


ROOT = Path(__file__).resolve().parents[1]


class RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.farm = load_farm(load_phase1_config(ROOT / "config" / "phase1.yaml"))

    def test_aerial_orientation_and_targets(self) -> None:
        east_west = aerial_parallel_sweep(self.farm, footprint_width_m=100, side_overlap=0.5)
        north_south = aerial_parallel_sweep(self.farm, footprint_width_m=100, side_overlap=0.5, orientation="north_south")
        self.assertTrue(all(item.start_y_m == item.end_y_m for item in east_west))
        self.assertTrue(all(item.start_x_m == item.end_x_m for item in north_south))
        targeted = aerial_parallel_sweep(self.farm, footprint_width_m=100, side_overlap=0.5, target_zone_ids=("Z_R1_C1",))
        self.assertTrue(all(max(item.start_x_m, item.end_x_m) <= 160 for item in targeted))

    def test_ground_patterns_are_deterministic_and_unique(self) -> None:
        for pattern in ("w", "serpentine"):
            route = ground_route(self.farm, pattern=pattern)
            self.assertEqual(len(route), len(set(route)))
        first = ground_route(self.farm, pattern="seeded_sample", sample_count=6, seed=4)
        self.assertEqual(first, ground_route(self.farm, pattern="seeded_sample", sample_count=6, seed=4))

    def test_explicit_route_rejects_unknown_zone(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            ground_route(self.farm, pattern="explicit", zone_ids=("missing",))


if __name__ == "__main__":
    unittest.main()
