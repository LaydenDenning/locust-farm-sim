"""Tests for Phase 7 drone and scout detection comparison."""

from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import fields, replace
from datetime import date
from pathlib import Path

from src.simulation.comparison import (
    MethodDetection,
    compare_methods,
    load_phase7_config,
)
from src.simulation.detection import DetectionRecord, Phase5Result
from src.simulation.run_phase7 import write_artifacts
from src.simulation.scouting import Phase6Result, ScoutObservation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "phase7.yaml"


def _drone_record(
    survey_id: str,
    survey_date: date,
    zone_id: str,
    status: str,
) -> DetectionRecord:
    available = status != "unavailable"
    return DetectionRecord(
        survey_id=survey_id,
        survey_date=survey_date,
        zone_id=zone_id,
        status=status,
        anomaly_score=0.8 if status == "flagged" else (0.1 if available else None),
        uncertainty=0.1 if available else None,
        quality_flag="good" if available else "missing",
        reason="test_input",
    )


def _scout_record(
    survey_id: str,
    survey_date: date,
    zone_id: str,
    score: float | None,
) -> ScoutObservation:
    return ScoutObservation(
        survey_id=survey_id,
        survey_date=survey_date,
        zone_id=zone_id,
        visit_order=1,
        visit_elapsed_minutes=4.0 if score is not None else None,
        visual_anomaly_score=score,
        uncertainty=0.2 if score is not None else None,
        quality_flag="good" if score is not None else "not_observed",
    )


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase7_config(CONFIG_PATH)
        cls.before_water = date(2022, 6, 5)
        cls.after_water = date(2022, 6, 19)
        cls.after_flood = date(2022, 7, 3)
        cls.drone = Phase5Result(
            detections=(
                _drone_record("SURVEY_001", cls.before_water, "Z_R2_C3", "flagged"),
                _drone_record("SURVEY_002", cls.after_water, "Z_R2_C3", "flagged"),
                _drone_record("SURVEY_002", cls.after_water, "Z_R2_C4", "clear"),
                _drone_record("SURVEY_002", cls.after_water, "Z_R1_C1", "flagged"),
                _drone_record("SURVEY_002", cls.after_water, "Z_R1_C2", "clear"),
                _drone_record("SURVEY_002", cls.after_water, "Z_R1_C3", "unavailable"),
            ),
            surveys=(),
        )
        cls.scout = Phase6Result(
            observations=(
                _scout_record("SURVEY_003", cls.after_flood, "Z_R3_C3", 0.8),
                _scout_record("SURVEY_003", cls.after_flood, "Z_R3_C2", 0.1),
                _scout_record("SURVEY_003", cls.after_flood, "Z_R5_C1", 0.8),
                _scout_record("SURVEY_003", cls.after_flood, "Z_R5_C3", None),
            ),
            surveys=(),
        )
        cls.result = compare_methods(cls.config, cls.drone, cls.scout)

    def test_config_uses_the_same_phase4_for_both_methods(self) -> None:
        self.assertEqual(
            self.config.phase5.phase4.source_path,
            self.config.phase6.phase4.source_path,
        )
        self.assertEqual(self.config.scout_rule.anomaly_threshold, 0.30)

    def test_truth_blind_method_records_have_no_issue_fields(self) -> None:
        names = {field.name for field in fields(MethodDetection)}
        self.assertFalse(
            names
            & {
                "issue_id",
                "mechanism",
                "truth_positive",
                "severity",
                "visibility",
                "yield_loss",
            }
        )

    def test_drone_and_scout_records_are_combined(self) -> None:
        self.assertEqual(len(self.result.method_detections), 10)
        self.assertEqual(
            {item.method for item in self.result.method_detections},
            {"drone", "scout"},
        )

    def test_evaluation_distinguishes_all_result_types(self) -> None:
        labels = {item.evaluation for item in self.result.evaluations}
        self.assertEqual(
            labels,
            {
                "true_positive",
                "false_positive",
                "false_negative",
                "true_negative",
                "unavailable",
            },
        )

    def test_pre_onset_flag_is_false_positive_and_not_first_detection(self) -> None:
        early = next(
            item
            for item in self.result.evaluations
            if item.method == "drone"
            and item.zone_id == "Z_R2_C3"
            and item.survey_date == self.before_water
        )
        self.assertEqual(early.evaluation, "false_positive")
        water = next(
            item
            for item in self.result.issue_summaries
            if item.issue_id == "WATER_01" and item.method == "drone"
        )
        self.assertEqual(water.first_detection_date, self.after_water)
        self.assertGreaterEqual(water.detection_delay_days, 0)

    def test_route_coverage_is_explicit_for_each_issue(self) -> None:
        water_scout = next(
            item
            for item in self.result.issue_summaries
            if item.issue_id == "WATER_01" and item.method == "scout"
        )
        flood_scout = next(
            item
            for item in self.result.issue_summaries
            if item.issue_id == "FLOOD_01" and item.method == "scout"
        )
        self.assertEqual(water_scout.observable_zone_count, 0)
        self.assertFalse(water_scout.detected)
        self.assertEqual(flood_scout.observable_zone_count, 1)
        self.assertTrue(flood_scout.detected)

    def test_every_issue_has_one_summary_per_method(self) -> None:
        issue_count = len(self.config.phase5.phase4.issues)
        self.assertEqual(len(self.result.issue_summaries), issue_count * 2)
        self.assertEqual(
            {(item.issue_id, item.method) for item in self.result.issue_summaries},
            {
                (issue.issue_id, method)
                for issue in self.config.phase5.phase4.issues
                for method in ("drone", "scout")
            },
        )

    def test_comparison_is_reproducible(self) -> None:
        self.assertEqual(
            self.result,
            compare_methods(self.config, self.drone, self.scout),
        )

    def test_duplicate_method_key_is_rejected(self) -> None:
        repeated = Phase5Result(
            detections=(self.drone.detections[0], self.drone.detections[0]),
            surveys=(),
        )
        with self.assertRaisesRegex(ValueError, "duplicate method detection key"):
            compare_methods(self.config, repeated, Phase6Result((), ()))

    def test_csv_output_counts_and_overwrite_protection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = replace(self.config.output, directory=Path(directory))
            config = replace(self.config, output=output)
            paths = write_artifacts(config, self.result)
            expected = (
                len(self.result.method_detections),
                len(self.result.evaluations),
                len(self.result.issue_summaries),
            )
            for path, count in zip(paths, expected, strict=True):
                with path.open(encoding="utf-8", newline="") as stream:
                    self.assertEqual(len(tuple(csv.DictReader(stream))), count)
            with self.assertRaises(FileExistsError):
                write_artifacts(config, self.result)


if __name__ == "__main__":
    unittest.main()
