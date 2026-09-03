"""Tests for Phase 10 provisional economics."""

from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.farm import load_farm
from src.simulation.actions import simulate_actions
from src.simulation.confirmation import Phase8Result
from src.simulation.economics import (
    EconomicsConfigError,
    STRATEGIES,
    calculate_economics,
    load_phase10_config,
)
from src.simulation.run_phase10 import write_artifacts


ROOT = Path(__file__).resolve().parents[1]


class EconomicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase10_config(ROOT / "config" / "phase10.yaml")
        phase1 = cls.config.phase9.phase8.phase7.phase5.phase4.phase1
        cls.farm = load_farm(phase1)
        cls.baseline = {zone.zone_id: 10_000.0 for zone in cls.farm.zones}
        cls.empty_phase8 = Phase8Result((), (), ())
        cls.empty_phase9 = simulate_actions(cls.config.phase9, cls.empty_phase8)

    def _calculate(self):
        return calculate_economics(
            self.config,
            self.empty_phase8,
            self.empty_phase9,
            farm=self.farm,
            baseline_twso_kg_ha=self.baseline,
        )

    def test_config_is_explicitly_proxy_economics(self) -> None:
        self.assertEqual(self.config.economics.currency, "USD")
        self.assertGreater(self.config.economics.twso_proxy_value_per_tonne, 0)

    def test_zone_rows_preserve_all_phase9_outcomes(self) -> None:
        result = self._calculate()
        self.assertEqual(
            len(result.zone_economics),
            len(self.empty_phase9.scenario_outcomes),
        )
        self.assertEqual(
            {item.strategy for item in result.zone_economics}, set(STRATEGIES)
        )

    def test_zone_area_and_twso_units_are_consistent(self) -> None:
        row = self._calculate().zone_economics[0]
        self.assertAlmostEqual(row.zone_area_ha, 2.56)
        expected = (
            row.baseline_twso_kg_ha
            * row.zone_area_ha
            * row.treated_loss_fraction
        )
        self.assertAlmostEqual(row.lost_twso_kg, expected)

    def test_no_intervention_has_no_operating_cost(self) -> None:
        summary = next(
            item for item in self._calculate().strategy_summary
            if item.strategy == "no_intervention"
        )
        self.assertEqual(summary.operating_cost, 0)
        self.assertEqual(summary.net_benefit_vs_no_intervention, 0)

    def test_drone_and_scout_costs_are_kept_separate(self) -> None:
        summaries = {item.strategy: item for item in self._calculate().strategy_summary}
        self.assertGreater(summaries["drone"].flight_cost, 0)
        self.assertGreater(summaries["drone"].processing_cost, 0)
        self.assertEqual(summaries["drone"].scouting_cost, 0)
        self.assertGreater(summaries["scout"].scouting_cost, 0)
        self.assertEqual(summaries["scout"].flight_cost, 0)

    def test_summary_cost_components_add_up(self) -> None:
        for item in self._calculate().strategy_summary:
            components = (
                item.flight_cost
                + item.processing_cost
                + item.scouting_cost
                + item.confirmation_cost
                + item.treatment_cost
                + item.false_positive_cost
            )
            self.assertAlmostEqual(item.operating_cost, components)
            self.assertAlmostEqual(
                item.total_cost, item.operating_cost + item.crop_loss_cost
            )

    def test_break_even_is_reported_only_for_drone(self) -> None:
        summaries = {item.strategy: item for item in self._calculate().strategy_summary}
        self.assertIsNotNone(summaries["drone"].break_even_drone_operations_cost)
        self.assertIsNone(summaries["scout"].break_even_drone_operations_cost)
        self.assertIsNone(
            summaries["no_intervention"].break_even_drone_operations_cost
        )

    def test_missing_or_invalid_baseline_is_rejected(self) -> None:
        missing = dict(self.baseline)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(ValueError, "keys"):
            calculate_economics(
                self.config,
                self.empty_phase8,
                self.empty_phase9,
                farm=self.farm,
                baseline_twso_kg_ha=missing,
            )
        invalid = dict(self.baseline)
        invalid[next(iter(invalid))] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            calculate_economics(
                self.config,
                self.empty_phase8,
                self.empty_phase9,
                farm=self.farm,
                baseline_twso_kg_ha=invalid,
            )

    def test_repeated_calculation_is_identical(self) -> None:
        self.assertEqual(self._calculate(), self._calculate())

    def test_invalid_cost_is_rejected(self) -> None:
        source = (ROOT / "config" / "phase10.yaml").read_text(encoding="utf-8")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "phase10.yaml"
            text = source.replace(
                "phase9_config: phase9.yaml",
                f"phase9_config: {str(ROOT / 'config' / 'phase9.yaml')}",
            ).replace("flight_cost_per_sortie: 20.0", "flight_cost_per_sortie: -1")
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(EconomicsConfigError):
                load_phase10_config(path)

    def test_csv_outputs_and_overwrite_protection(self) -> None:
        result = self._calculate()
        with TemporaryDirectory() as directory:
            output = type(self.config.output)(
                Path(directory), "zones.csv", "summary.csv"
            )
            config = type(self.config)(
                self.config.source_path,
                self.config.phase9,
                self.config.economics,
                output,
            )
            paths = write_artifacts(config, result)
            self.assertEqual(len(paths), 2)
            with paths[0].open(encoding="utf-8", newline="") as stream:
                self.assertEqual(
                    len(tuple(csv.DictReader(stream))), len(result.zone_economics)
                )
            with self.assertRaises(FileExistsError):
                write_artifacts(config, result)


if __name__ == "__main__":
    unittest.main()
