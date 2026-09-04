"""Tests for paired modular monitoring experiments."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import unittest

import pandas as pd

from src.farm import load_farm
from src.simulation.experiment import run_experiment
from src.simulation.profiles import load_experiment_config
from src.simulation.scenarios import load_phase11_config, run_scenarios


ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base = load_experiment_config(ROOT / "config" / "experiments" / "baseline.yaml")
        cls.config = replace(base, scenario_count=4)
        cls.farm = load_farm(base.phase1)
        cls.baseline = {zone.zone_id: 10_000.0 for zone in cls.farm.zones}
        truth_rows = []
        for day in base.methods[0].schedule.days:
            survey_date = base.phase1.calendar.base_sowing_date + timedelta(days=day)
            for zone in cls.farm.zones:
                truth_rows.append({"zone_id": zone.zone_id, "date": survey_date, "crop_active": True, "LAI": 3.0, "NNI": 0.9, "SM": 0.28, "soil_smw": 0.10, "soil_smfcf": 0.30})
        cls.truth = pd.DataFrame(truth_rows)

    def _run(self, config=None):
        return run_experiment(config or self.config, self.truth, farm=self.farm, baseline_twso_kg_ha=self.baseline)

    def test_fixed_seed_is_reproducible(self) -> None:
        self.assertEqual(self._run(), self._run())

    def test_variant_order_does_not_change_results(self) -> None:
        first = self._run()
        second = self._run(replace(self.config, methods=tuple(reversed(self.config.methods))))
        key = lambda row: (row.scenario_id, row.method_id)
        self.assertEqual(sorted(first.method_results, key=key), sorted(second.method_results, key=key))

    def test_long_form_keys_and_pairwise_deltas_reconcile(self) -> None:
        result = self._run()
        keys = {(row.scenario_id, row.method_id) for row in result.method_results}
        self.assertEqual(len(keys), len(result.method_results))
        indexed = {(row.scenario_id, row.method_id): row for row in result.method_results}
        for row in result.pairwise_results:
            if row.reference_method_id == "no_intervention":
                self.assertAlmostEqual(row.net_benefit_delta, indexed[(row.scenario_id, row.candidate_method_id)].net_benefit_vs_no_intervention)
            else:
                expected = indexed[(row.scenario_id, row.candidate_method_id)].net_benefit_vs_no_intervention - indexed[(row.scenario_id, row.reference_method_id)].net_benefit_vs_no_intervention
                self.assertAlmostEqual(row.net_benefit_delta, expected)

    def test_compatibility_bridge_matches_phase11_outcomes(self) -> None:
        result = self._run()
        legacy_config = load_phase11_config(ROOT / "config" / "phase11.yaml")
        legacy_config = replace(legacy_config, scenarios=replace(legacy_config.scenarios, count=self.config.scenario_count))
        legacy = run_scenarios(legacy_config, self.truth, farm=self.farm, baseline_twso_kg_ha=self.baseline)
        modular = {(row.scenario_id, row.method_kind): row for row in result.method_results}
        for row in legacy.scenarios:
            self.assertEqual(modular[(row.scenario_id, "drone")].net_benefit_vs_no_intervention, row.drone_net_benefit)
            self.assertEqual(modular[(row.scenario_id, "ground_scout")].avoided_twso_kg, row.scout_avoided_twso_kg)

    def test_no_issue_scenarios_retain_operating_cost(self) -> None:
        scenario = replace(self.config.scenario_profile, no_issue_probability=1.0)
        result = self._run(replace(self.config, scenario_profile=scenario, scenario_count=2, compatibility_phase11_config=None))
        self.assertTrue(all(row.mechanism == "none" for row in result.method_results))
        self.assertTrue(all(row.avoided_twso_kg == 0 for row in result.method_results))
        self.assertTrue(all(row.net_benefit_vs_no_intervention <= 0 for row in result.method_results))

    def test_four_method_spatial_experiment_runs(self) -> None:
        config = load_experiment_config(ROOT / "config" / "experiments" / "four_method_comparison.yaml")
        result = self._run(replace(config, scenario_count=2))
        self.assertEqual({row.method_kind for row in result.method_results}, {"drone", "ground_scout", "manned_aircraft", "satellite"})


if __name__ == "__main__":
    unittest.main()
