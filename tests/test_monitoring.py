"""Tests for common monitoring adapters."""

from pathlib import Path
import unittest

from src.farm import load_farm
from src.simulation.monitoring import METHOD_RUNNERS, run_method
from src.simulation.profiles import load_experiment_config
from src.simulation.spatial import build_analysis_grid


ROOT = Path(__file__).resolve().parents[1]


class MonitoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_experiment_config(ROOT / "config" / "experiments" / "four_method_comparison.yaml")
        cls.farm = load_farm(cls.config.phase1)
        cls.cells = build_analysis_grid(cls.farm, 10)

    def test_all_four_method_kinds_are_registered(self) -> None:
        self.assertTrue({"drone", "ground_scout", "manned_aircraft", "satellite"} <= set(METHOD_RUNNERS))

    def test_method_results_use_canonical_zone_rows(self) -> None:
        fractions = {zone.zone_id: 0.0 for zone in self.farm.zones}
        for variant in self.config.methods:
            result = run_method(variant, farm=self.farm, cells=self.cells, crop=self.config.crop_profile, issue=None, affected_fractions=fractions, scenario_seed=10)
            self.assertEqual(result.method_id, variant.method_id)
            self.assertEqual(len(result.observations), len(variant.schedule.days) * len(self.farm.zones))
            self.assertGreaterEqual(result.operating_cost, 0)

    def test_keyed_randomness_is_shared_by_family(self) -> None:
        frequency = load_experiment_config(ROOT / "config" / "experiments" / "drone_frequency.yaml")
        fractions = {zone.zone_id: 0.0 for zone in self.farm.zones}
        results = [run_method(item, farm=self.farm, cells=self.cells, crop=frequency.crop_profile, issue=None, affected_fractions=fractions, scenario_seed=22) for item in frequency.methods[:2]]
        first = {(row.observed_date, row.zone_id): row.quality_flag for row in results[0].observations}
        second = {(row.observed_date, row.zone_id): row.quality_flag for row in results[1].observations}
        shared = set(first) & set(second)
        self.assertTrue(shared)
        self.assertTrue(all(first[key] == second[key] for key in shared))


if __name__ == "__main__":
    unittest.main()
