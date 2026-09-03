"""Tests for Phase 11 repeated paired scenarios."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.farm import load_farm
from src.simulation.issues import ISSUE_MECHANISMS
from src.simulation.run_phase11 import write_artifacts
from src.simulation.scenarios import (
    ScenarioConfigError,
    load_phase11_config,
    run_scenarios,
)


ROOT = Path(__file__).resolve().parents[1]


class ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base = load_phase11_config(ROOT / "config" / "phase11.yaml")
        cls.config = replace(base, scenarios=replace(base.scenarios, count=6))
        phase1 = base.phase10.phase9.phase8.phase7.phase5.phase4.phase1
        cls.farm = load_farm(phase1)
        cls.baseline = {zone.zone_id: 10_000.0 for zone in cls.farm.zones}
        rows = []
        for day in base.phase10.phase9.phase8.phase7.phase5.phase4.schedule.survey_days:
            date = phase1.calendar.base_sowing_date + timedelta(days=day)
            for zone in cls.farm.zones:
                rows.append({
                    "zone_id": zone.zone_id,
                    "date": date,
                    "crop_active": True,
                    "LAI": 3.0,
                    "NNI": 0.9,
                    "SM": 0.28,
                    "soil_smw": 0.10,
                    "soil_smfcf": 0.30,
                })
        cls.truth = pd.DataFrame(rows)

    def _run(self, config=None):
        return run_scenarios(
            config or self.config,
            self.truth,
            farm=self.farm,
            baseline_twso_kg_ha=self.baseline,
        )

    def test_fixed_seed_is_reproducible(self) -> None:
        self.assertEqual(self._run(), self._run())

    def test_configured_number_of_paired_results_is_returned(self) -> None:
        result = self._run()
        self.assertEqual(len(result.scenarios), self.config.scenarios.count)
        self.assertTrue(all(item.scenario_id for item in result.scenarios))
        self.assertTrue(all(item.scenario_seed >= 0 for item in result.scenarios))

    def test_sampled_issues_use_supported_mechanisms_and_ranges(self) -> None:
        settings = self.config.scenarios
        for item in self._run().scenarios:
            self.assertIn(item.mechanism, set(ISSUE_MECHANISMS) | {"none"})
            if item.mechanism != "none":
                self.assertLessEqual(settings.footprint_zone_count.minimum, item.footprint_zone_count)
                self.assertLessEqual(item.footprint_zone_count, settings.footprint_zone_count.maximum)
                self.assertLessEqual(settings.onset_day.minimum, item.onset_day)
                self.assertLessEqual(item.onset_day, settings.onset_day.maximum)

    def test_sampled_footprints_are_contiguous(self) -> None:
        positions = {zone.zone_id: (zone.row, zone.column) for zone in self.farm.zones}
        for item in self._run().scenarios:
            remaining = set(filter(None, item.zone_ids.split("|")))
            if not remaining:
                continue
            reached = {remaining.pop()}
            while remaining:
                adjacent = {
                    candidate for candidate in remaining
                    if any(
                        abs(positions[candidate][0] - positions[known][0])
                        + abs(positions[candidate][1] - positions[known][1]) == 1
                        for known in reached
                    )
                }
                self.assertTrue(adjacent)
                reached.update(adjacent)
                remaining.difference_update(adjacent)

    def test_no_issue_scenarios_capture_operating_cost_without_crop_benefit(self) -> None:
        settings = replace(self.config.scenarios, count=2, no_issue_probability=1.0)
        result = self._run(replace(self.config, scenarios=settings))
        self.assertTrue(all(item.mechanism == "none" for item in result.scenarios))
        self.assertTrue(all(item.drone_avoided_twso_kg == 0 for item in result.scenarios))
        self.assertTrue(all(item.drone_net_benefit < 0 for item in result.scenarios))

    def test_distribution_summaries_are_bounded_and_complete(self) -> None:
        result = self._run()
        self.assertEqual(len(result.distributions), 5)
        for item in result.distributions:
            self.assertEqual(item.sample_count, self.config.scenarios.count)
            self.assertLessEqual(item.percentile_05, item.median)
            self.assertLessEqual(item.median, item.percentile_95)
            self.assertGreaterEqual(item.probability_above_zero, 0)
            self.assertLessEqual(item.probability_above_zero, 1)

    def test_sensitivity_correlations_are_finite_and_bounded(self) -> None:
        result = self._run()
        self.assertGreater(len(result.sensitivities), 0)
        for item in result.sensitivities:
            self.assertGreaterEqual(item.correlation_with_drone_advantage, -1)
            self.assertLessEqual(item.correlation_with_drone_advantage, 1)

    def test_invalid_range_is_rejected(self) -> None:
        source = (ROOT / "config" / "phase11.yaml").read_text(encoding="utf-8")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "phase11.yaml"
            text = source.replace(
                "phase10_config: phase10.yaml",
                f"phase10_config: {str(ROOT / 'config' / 'phase10.yaml')}",
            ).replace("footprint_zone_count: [1, 4]", "footprint_zone_count: [4, 1]")
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(ScenarioConfigError):
                load_phase11_config(path)

    def test_csv_outputs_and_overwrite_protection(self) -> None:
        result = self._run()
        with TemporaryDirectory() as directory:
            output = type(self.config.output)(
                Path(directory), "scenarios.csv", "distributions.csv", "sensitivity.csv"
            )
            config = replace(self.config, output=output)
            paths = write_artifacts(config, result)
            self.assertEqual(len(paths), 3)
            with paths[0].open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(tuple(csv.DictReader(stream))), len(result.scenarios))
            with self.assertRaises(FileExistsError):
                write_artifacts(config, result)


if __name__ == "__main__":
    unittest.main()
