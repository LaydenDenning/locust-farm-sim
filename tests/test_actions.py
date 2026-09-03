"""Tests for Phase 9 treatment actions and paired outcomes."""

from __future__ import annotations

import csv
from dataclasses import fields
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.simulation.actions import (
    ActionConfigError,
    ActionEvent,
    STRATEGIES,
    load_phase9_config,
    simulate_actions,
)
from src.simulation.confirmation import (
    ConfirmationEvaluation,
    ConfirmationEvent,
    Phase8Result,
)
from src.simulation.run_phase9 import write_artifacts


ROOT = Path(__file__).resolve().parents[1]


class ActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase9_config(ROOT / "config" / "phase9.yaml")
        cls.sowing_date = (
            cls.config.phase8.phase7.phase5.phase4.phase1.calendar.base_sowing_date
        )

    def _phase8_result(
        self,
        *records: tuple[str, str, str, int, str, bool],
    ) -> Phase8Result:
        events = []
        evaluations = []
        for index, (method, zone_id, result, day, evaluation, truth) in enumerate(
            records, start=1
        ):
            confirmation_id = f"CONF_{index:04d}"
            confirmation_date = self.sowing_date + timedelta(days=day)
            events.append(
                ConfirmationEvent(
                    confirmation_id=confirmation_id,
                    method=method,
                    survey_id=f"SURVEY_{index:03d}",
                    zone_id=zone_id,
                    flag_date=confirmation_date - timedelta(days=1),
                    confirmation_date=confirmation_date,
                    result=result,
                )
            )
            evaluations.append(
                ConfirmationEvaluation(
                    confirmation_id=confirmation_id,
                    method=method,
                    survey_id=f"SURVEY_{index:03d}",
                    zone_id=zone_id,
                    flag_date=confirmation_date - timedelta(days=1),
                    confirmation_date=confirmation_date,
                    truth_positive=truth,
                    evaluation=evaluation,
                )
            )
        return Phase8Result(tuple(events), tuple(evaluations), ())

    def test_config_has_one_rule_per_mechanism(self) -> None:
        rules = {rule.mechanism: rule for rule in self.config.intervention_rules}
        self.assertEqual(
            set(rules),
            {"water_deficit", "excess_water", "nutrient_deficit", "canopy_damage"},
        )
        self.assertTrue(all(rule.cost_per_ha >= 0 for rule in rules.values()))

    def test_only_confirmed_findings_create_actions(self) -> None:
        phase8 = self._phase8_result(
            ("drone", "Z_R2_C3", "confirmed", 50, "true_confirmation", True),
            ("scout", "Z_R3_C3", "rejected", 65, "missed_confirmation", True),
        )
        result = simulate_actions(self.config, phase8)
        self.assertEqual(len(result.action_events), 1)
        self.assertEqual(result.action_events[0].confirmation_id, "CONF_0001")

    def test_repeated_confirmations_do_not_repeat_treatment(self) -> None:
        phase8 = self._phase8_result(
            ("drone", "Z_R2_C3", "confirmed", 50, "true_confirmation", True),
            ("drone", "Z_R2_C3", "confirmed", 57, "true_confirmation", True),
        )
        result = simulate_actions(self.config, phase8)
        self.assertEqual(len(result.action_events), 1)
        self.assertEqual(result.action_events[0].confirmation_id, "CONF_0001")

    def test_action_never_precedes_confirmation(self) -> None:
        phase8 = self._phase8_result(
            ("drone", "Z_R2_C3", "confirmed", 50, "true_confirmation", True),
        )
        action = simulate_actions(self.config, phase8).action_events[0]
        self.assertEqual(action.action_date, action.confirmation_date)
        with self.assertRaisesRegex(ValueError, "must not precede"):
            ActionEvent(
                "A", "drone", "C", "S", "Z", action.confirmation_date,
                action.confirmation_date - timedelta(days=1), "treat"
            )

    def test_operational_action_has_no_truth_fields(self) -> None:
        names = {item.name for item in fields(ActionEvent)}
        self.assertNotIn("truth_positive", names)
        self.assertNotIn("issue_id", names)
        self.assertNotIn("mechanism", names)
        self.assertNotIn("yield", names)

    def test_false_confirmation_creates_unnecessary_action_only(self) -> None:
        phase8 = self._phase8_result(
            ("drone", "Z_R1_C1", "confirmed", 50, "false_confirmation", False),
        )
        result = simulate_actions(self.config, phase8)
        self.assertEqual(result.action_evaluations[0].evaluation, "unnecessary_action")
        self.assertTrue(
            all(item.action_date is None for item in result.scenario_outcomes)
        )

    def test_three_paired_outcomes_exist_for_every_issue_zone(self) -> None:
        result = simulate_actions(self.config, self._phase8_result())
        issue_zone_count = sum(
            len(issue.zone_ids)
            for issue in self.config.phase8.phase7.phase5.phase4.issues
        )
        self.assertEqual(len(result.scenario_outcomes), issue_zone_count * 3)
        grouped = {}
        for item in result.scenario_outcomes:
            grouped.setdefault((item.issue_id, item.zone_id), set()).add(item.strategy)
        self.assertTrue(all(value == set(STRATEGIES) for value in grouped.values()))

    def test_no_intervention_preserves_full_untreated_loss(self) -> None:
        result = simulate_actions(self.config, self._phase8_result())
        untreated = [
            item for item in result.scenario_outcomes
            if item.strategy == "no_intervention"
        ]
        self.assertTrue(all(item.action_date is None for item in untreated))
        self.assertTrue(
            all(
                item.treated_loss_fraction == item.untreated_loss_fraction
                and item.avoided_loss_fraction == 0
                for item in untreated
            )
        )

    def test_true_confirmation_reduces_only_matching_zone_and_method(self) -> None:
        phase8 = self._phase8_result(
            ("drone", "Z_R2_C3", "confirmed", 50, "true_confirmation", True),
        )
        result = simulate_actions(self.config, phase8)
        matching = next(
            item for item in result.scenario_outcomes
            if item.issue_id == "WATER_01"
            and item.zone_id == "Z_R2_C3"
            and item.strategy == "drone"
        )
        same_issue_other_zone = next(
            item for item in result.scenario_outcomes
            if item.issue_id == "WATER_01"
            and item.zone_id == "Z_R2_C4"
            and item.strategy == "drone"
        )
        scout = next(
            item for item in result.scenario_outcomes
            if item.issue_id == "WATER_01"
            and item.zone_id == "Z_R2_C3"
            and item.strategy == "scout"
        )
        self.assertTrue(matching.action_effective)
        self.assertGreater(matching.avoided_loss_fraction, 0)
        self.assertEqual(same_issue_other_zone.avoided_loss_fraction, 0)
        self.assertEqual(scout.avoided_loss_fraction, 0)

    def test_late_action_completes_without_biological_benefit(self) -> None:
        phase8 = self._phase8_result(
            ("drone", "Z_R3_C5", "confirmed", 100, "true_confirmation", True),
        )
        outcome = next(
            item for item in simulate_actions(self.config, phase8).scenario_outcomes
            if item.issue_id == "NUTRIENT_01"
            and item.zone_id == "Z_R3_C5"
            and item.strategy == "drone"
        )
        self.assertFalse(outcome.action_effective)
        self.assertEqual(outcome.avoided_loss_fraction, 0)
        self.assertGreater(outcome.action_cost_per_ha, 0)

    def test_results_are_reproducible(self) -> None:
        phase8 = self._phase8_result(
            ("drone", "Z_R2_C3", "confirmed", 50, "true_confirmation", True),
            ("scout", "Z_R3_C3", "confirmed", 65, "true_confirmation", True),
        )
        self.assertEqual(
            simulate_actions(self.config, phase8),
            simulate_actions(self.config, phase8),
        )

    def test_invalid_config_is_rejected(self) -> None:
        text = (ROOT / "config" / "phase9.yaml").read_text(encoding="utf-8")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "phase9.yaml"
            path.write_text(
                text.replace("    efficacy: 0.80", "    efficacy: 1.20"),
                encoding="utf-8",
            )
            # Keep the parent configuration reference valid in the temporary copy.
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "phase8_config: phase8.yaml",
                    f"phase8_config: {str(ROOT / 'config' / 'phase8.yaml')}",
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ActionConfigError):
                load_phase9_config(path)

    def test_csv_outputs_and_overwrite_protection(self) -> None:
        phase8 = self._phase8_result(
            ("drone", "Z_R2_C3", "confirmed", 50, "true_confirmation", True),
        )
        result = simulate_actions(self.config, phase8)
        with TemporaryDirectory() as directory:
            output = type(self.config.output)(
                Path(directory), "actions.csv", "evaluation.csv", "outcomes.csv"
            )
            config = type(self.config)(
                self.config.source_path,
                self.config.phase8,
                self.config.intervention_rules,
                output,
            )
            paths = write_artifacts(config, result)
            self.assertEqual(len(paths), 3)
            with paths[0].open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(tuple(csv.DictReader(stream))), 1)
            with paths[2].open(encoding="utf-8", newline="") as stream:
                self.assertEqual(
                    len(tuple(csv.DictReader(stream))),
                    len(result.scenario_outcomes),
                )
            with self.assertRaises(FileExistsError):
                write_artifacts(config, result)


if __name__ == "__main__":
    unittest.main()
