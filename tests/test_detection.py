"""Tests for Phase 5 sensor-only anomaly classification."""

from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import fields, replace
from datetime import date
from pathlib import Path

from src.simulation.detection import (
    DetectionRecord,
    DetectionRule,
    classify_observations,
    load_phase5_config,
)
from src.simulation.observations import Observation
from src.simulation.run_phase5 import write_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "phase5.yaml"


def _observation(
    zone_id: str,
    *,
    survey_id: str = "SURVEY_001",
    anomaly: float | None = 0.1,
    uncertainty: float | None = 0.1,
    quality: str = "good",
    covered: bool = True,
) -> Observation:
    available = anomaly is not None and uncertainty is not None
    return Observation(
        survey_id=survey_id,
        survey_date=date(2022, 6, 5),
        zone_id=zone_id,
        covered=covered,
        coverage_fraction=1.0 if covered else 0.0,
        crop_active=True,
        relative_red=0.1 if available else None,
        relative_green=0.3 if available else None,
        relative_blue=0.1 if available else None,
        relative_nir=0.7 if available else None,
        ndvi_like=0.75 if available else None,
        canopy_cover=0.8 if available else None,
        anomaly_score=anomaly,
        uncertainty=uncertainty,
        quality_flag=quality,
    )


class DetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase5_config(CONFIG_PATH)

    def test_config_references_phase4_and_strict_rule(self) -> None:
        self.assertEqual(self.config.rule.anomaly_threshold, 0.30)
        self.assertEqual(self.config.rule.maximum_uncertainty, 0.25)
        self.assertEqual(self.config.rule.allowed_quality_flags, ("good",))
        self.assertTrue(self.config.phase4.issues_file.is_file())

    def test_classifies_flagged_clear_and_unavailable(self) -> None:
        observations = (
            _observation("Z_R1_C1", anomaly=0.30),
            _observation("Z_R1_C2", anomaly=0.29),
            _observation("Z_R1_C3", covered=False),
            _observation("Z_R1_C4", anomaly=None, uncertainty=None, quality="missing"),
            _observation("Z_R1_C5", anomaly=0.8, uncertainty=0.5),
            _observation("Z_R2_C1", anomaly=0.8, quality="cloud"),
        )
        result = classify_observations(observations, self.config.rule)
        statuses = {record.zone_id: record.status for record in result.detections}
        self.assertEqual(statuses["Z_R1_C1"], "flagged")
        self.assertEqual(statuses["Z_R1_C2"], "clear")
        self.assertEqual(statuses["Z_R1_C3"], "unavailable")
        self.assertEqual(statuses["Z_R1_C4"], "unavailable")
        self.assertEqual(statuses["Z_R1_C5"], "unavailable")
        self.assertEqual(statuses["Z_R2_C1"], "unavailable")

    def test_summary_counts_match_detection_records(self) -> None:
        observations = (
            _observation("Z_R1_C1", anomaly=0.5),
            _observation("Z_R1_C2", anomaly=0.1),
            _observation("Z_R1_C3", covered=False),
        )
        summary = classify_observations(observations, self.config.rule).surveys[0]
        self.assertEqual(summary.total_zones, 3)
        self.assertEqual(summary.available_zones, 2)
        self.assertEqual(summary.flagged_zones, 1)
        self.assertEqual(summary.clear_zones, 1)
        self.assertEqual(summary.unavailable_zones, 1)

    def test_detection_output_contains_no_hidden_truth_fields(self) -> None:
        names = {field.name for field in fields(DetectionRecord)}
        self.assertFalse(
            names
            & {
                "issue_id",
                "mechanism",
                "severity",
                "visibility",
                "untreated_loss_fraction",
                "yield_loss",
            }
        )

    def test_duplicate_observation_key_is_rejected(self) -> None:
        observation = _observation("Z_R1_C1")
        with self.assertRaisesRegex(ValueError, "duplicate observation key"):
            classify_observations((observation, observation), self.config.rule)

    def test_rule_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            DetectionRule(1.1, 0.2, ("good",))
        with self.assertRaises(ValueError):
            DetectionRule(0.3, -0.1, ("good",))
        with self.assertRaises(ValueError):
            DetectionRule(0.3, 0.2, ())

    def test_same_observations_produce_identical_results(self) -> None:
        observations = (
            _observation("Z_R1_C1", anomaly=0.5),
            _observation("Z_R1_C2", anomaly=0.1),
        )
        first = classify_observations(observations, self.config.rule)
        second = classify_observations(observations, self.config.rule)
        self.assertEqual(first, second)

    def test_csv_output_counts_and_overwrite_protection(self) -> None:
        result = classify_observations(
            (_observation("Z_R1_C1", anomaly=0.5),), self.config.rule
        )
        with tempfile.TemporaryDirectory() as directory:
            output = replace(self.config.output, directory=Path(directory))
            config = replace(self.config, output=output)
            paths = write_artifacts(config, result)
            with paths[0].open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(tuple(csv.DictReader(stream))), 1)
            with paths[1].open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(tuple(csv.DictReader(stream))), 1)
            with self.assertRaises(FileExistsError):
                write_artifacts(config, result)


if __name__ == "__main__":
    unittest.main()
