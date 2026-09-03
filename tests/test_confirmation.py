"""Tests for Phase 8 human confirmation."""

from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import fields, replace
from datetime import date
from pathlib import Path

from src.simulation.comparison import (
    DetectionEvaluation,
    MethodDetection,
    Phase7Result,
)
from src.simulation.confirmation import (
    ConfirmationEvent,
    ConfirmationSettings,
    load_phase8_config,
    simulate_confirmations,
)
from src.simulation.run_phase8 import write_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "phase8.yaml"


def _method_detection(
    method: str,
    survey_id: str,
    survey_date: date,
    zone_id: str,
    status: str = "flagged",
) -> MethodDetection:
    return MethodDetection(
        method=method,
        survey_id=survey_id,
        survey_date=survey_date,
        zone_id=zone_id,
        status=status,
        score=0.8 if status == "flagged" else 0.1,
        uncertainty=0.1,
        quality_flag="good",
        reason="test_input",
    )


def _evaluation(item: MethodDetection, truth_positive: bool) -> DetectionEvaluation:
    if item.status == "flagged":
        label = "true_positive" if truth_positive else "false_positive"
    else:
        label = "false_negative" if truth_positive else "true_negative"
    return DetectionEvaluation(
        method=item.method,
        survey_id=item.survey_id,
        survey_date=item.survey_date,
        zone_id=item.zone_id,
        status=item.status,
        truth_positive=truth_positive,
        evaluation=label,
    )


class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase8_config(CONFIG_PATH)
        water = _method_detection(
            "drone", "SURVEY_002", date(2022, 6, 19), "Z_R2_C3"
        )
        flood = _method_detection(
            "scout", "SURVEY_003", date(2022, 7, 3), "Z_R3_C3"
        )
        false_flag = _method_detection(
            "drone", "SURVEY_002", date(2022, 6, 19), "Z_R1_C1"
        )
        clear = _method_detection(
            "drone", "SURVEY_002", date(2022, 6, 19), "Z_R2_C4", "clear"
        )
        cls.phase7 = Phase7Result(
            method_detections=(water, flood, false_flag, clear),
            evaluations=(
                _evaluation(water, True),
                _evaluation(flood, True),
                _evaluation(false_flag, False),
                _evaluation(clear, True),
            ),
            issue_summaries=(),
        )

    def test_config_has_method_specific_delays(self) -> None:
        settings = self.config.confirmation
        self.assertEqual(settings.delay_for("drone"), 1)
        self.assertEqual(settings.delay_for("scout"), 0)
        self.assertEqual(settings.sensitivity, 0.95)
        self.assertEqual(settings.specificity, 0.98)

    def test_only_flags_create_confirmation_requests(self) -> None:
        config = replace(
            self.config,
            confirmation=replace(
                self.config.confirmation, sensitivity=1.0, specificity=1.0
            ),
        )
        result = simulate_confirmations(config, self.phase7)
        self.assertEqual(len(result.confirmation_events), 3)
        self.assertNotIn(
            "Z_R2_C4", {item.zone_id for item in result.confirmation_events}
        )

    def test_drone_is_delayed_and_scout_is_same_day(self) -> None:
        config = replace(
            self.config,
            confirmation=replace(
                self.config.confirmation, sensitivity=1.0, specificity=1.0
            ),
        )
        result = simulate_confirmations(config, self.phase7)
        drone = next(
            item
            for item in result.confirmation_events
            if item.method == "drone" and item.zone_id == "Z_R2_C3"
        )
        scout = next(
            item
            for item in result.confirmation_events
            if item.method == "scout"
        )
        self.assertEqual((drone.confirmation_date - drone.flag_date).days, 1)
        self.assertEqual(scout.confirmation_date, scout.flag_date)

    def test_perfect_confirmation_accepts_truth_and_rejects_false_flag(self) -> None:
        config = replace(
            self.config,
            confirmation=replace(
                self.config.confirmation, sensitivity=1.0, specificity=1.0
            ),
        )
        result = simulate_confirmations(config, self.phase7)
        labels = {item.evaluation for item in result.evaluations}
        self.assertEqual(labels, {"true_confirmation", "correct_rejection"})
        false_event = next(
            item
            for item in result.confirmation_events
            if item.zone_id == "Z_R1_C1"
        )
        self.assertEqual(false_event.result, "rejected")

    def test_imperfect_confirmation_labels_both_error_types(self) -> None:
        config = replace(
            self.config,
            confirmation=replace(
                self.config.confirmation, sensitivity=0.0, specificity=0.0
            ),
        )
        result = simulate_confirmations(config, self.phase7)
        labels = {item.evaluation for item in result.evaluations}
        self.assertIn("missed_confirmation", labels)
        self.assertIn("false_confirmation", labels)

    def test_issue_summary_uses_first_true_confirmation(self) -> None:
        config = replace(
            self.config,
            confirmation=replace(
                self.config.confirmation, sensitivity=1.0, specificity=1.0
            ),
        )
        result = simulate_confirmations(config, self.phase7)
        water = next(
            item
            for item in result.issue_summaries
            if item.issue_id == "WATER_01" and item.method == "drone"
        )
        flood = next(
            item
            for item in result.issue_summaries
            if item.issue_id == "FLOOD_01" and item.method == "scout"
        )
        self.assertTrue(water.confirmed)
        self.assertEqual(water.first_confirmation_date, date(2022, 6, 20))
        self.assertEqual(water.confirmation_delay_days, 5)
        self.assertTrue(flood.confirmed)
        self.assertEqual(flood.first_confirmation_date, date(2022, 7, 3))

    def test_operational_event_has_no_truth_fields(self) -> None:
        names = {field.name for field in fields(ConfirmationEvent)}
        self.assertFalse(
            names
            & {
                "truth_positive",
                "evaluation",
                "issue_id",
                "mechanism",
                "severity",
                "yield_loss",
            }
        )

    def test_seeded_confirmation_is_reproducible(self) -> None:
        first = simulate_confirmations(self.config, self.phase7)
        second = simulate_confirmations(self.config, self.phase7)
        self.assertEqual(first, second)

    def test_invalid_settings_are_rejected(self) -> None:
        values = self.config.confirmation
        with self.assertRaises(ValueError):
            ConfirmationSettings(values.seed, -1, 0, 0.9, 0.9)
        with self.assertRaises(ValueError):
            replace(values, sensitivity=1.1)
        with self.assertRaises(ValueError):
            replace(values, specificity=-0.1)

    def test_csv_output_counts_and_overwrite_protection(self) -> None:
        result = simulate_confirmations(self.config, self.phase7)
        with tempfile.TemporaryDirectory() as directory:
            output = replace(self.config.output, directory=Path(directory))
            config = replace(self.config, output=output)
            paths = write_artifacts(config, result)
            expected = (
                len(result.confirmation_events),
                len(result.evaluations),
                len(result.issue_summaries),
            )
            for path, count in zip(paths, expected, strict=True):
                with path.open(encoding="utf-8", newline="") as stream:
                    self.assertEqual(len(tuple(csv.DictReader(stream))), count)
            with self.assertRaises(FileExistsError):
                write_artifacts(config, result)


if __name__ == "__main__":
    unittest.main()
