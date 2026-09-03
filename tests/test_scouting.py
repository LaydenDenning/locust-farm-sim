"""Tests for Phase 6 conventional scout observations."""

from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

from src.simulation.run_phase6 import write_artifacts
from src.simulation.scouting import (
    ScoutObservation,
    ScoutSettings,
    load_phase6_config,
    simulate_scouting,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "phase6.yaml"


class ScoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase6_config(CONFIG_PATH)
        cls.result = simulate_scouting(cls.config)

    def test_config_has_nine_unique_w_route_zones(self) -> None:
        route = self.config.scout.route_zone_ids
        self.assertEqual(len(route), 9)
        self.assertEqual(len(set(route)), 9)
        self.assertEqual(route[0], "Z_R5_C1")
        self.assertEqual(route[-1], "Z_R5_C5")

    def test_scout_uses_same_weekly_schedule_as_phase4(self) -> None:
        expected_surveys = len(self.config.phase4.schedule.survey_days)
        self.assertEqual(len(self.result.surveys), expected_surveys)
        self.assertEqual(len(self.result.observations), expected_surveys * 9)
        self.assertEqual(
            tuple(item.campaign_day for item in self.result.surveys),
            self.config.phase4.schedule.survey_days,
        )

    def test_visit_order_and_duration_are_consistent(self) -> None:
        first_survey = self.result.surveys[0]
        observations = [
            item
            for item in self.result.observations
            if item.survey_id == first_survey.survey_id
        ]
        self.assertEqual([item.visit_order for item in observations], list(range(1, 10)))
        self.assertAlmostEqual(first_survey.route_duration_minutes, 60.0)
        self.assertAlmostEqual(observations[-1].visit_elapsed_minutes, 60.0)

    def test_observations_are_bounded_and_reproducible(self) -> None:
        self.assertEqual(self.result, simulate_scouting(self.config))
        for observation in self.result.observations:
            if observation.visual_anomaly_score is not None:
                self.assertGreaterEqual(observation.visual_anomaly_score, 0.0)
                self.assertLessEqual(observation.visual_anomaly_score, 1.0)
            if observation.uncertainty is not None:
                self.assertGreaterEqual(observation.uncertainty, 0.0)
                self.assertLessEqual(observation.uncertainty, 1.0)

    def test_visible_issue_increases_visual_score_without_noise(self) -> None:
        scout = replace(
            self.config.scout,
            visual_noise_std=0.0,
            missing_probability=0.0,
        )
        result = simulate_scouting(replace(self.config, scout=scout))
        final_survey = result.surveys[-1].survey_id
        affected = next(
            item
            for item in result.observations
            if item.survey_id == final_survey and item.zone_id == "Z_R3_C3"
        )
        healthy = next(
            item
            for item in result.observations
            if item.survey_id == final_survey and item.zone_id == "Z_R3_C2"
        )
        self.assertGreater(affected.visual_anomaly_score, healthy.visual_anomaly_score)

    def test_missed_survey_preserves_planned_rows_as_unavailable(self) -> None:
        missed_day = self.config.phase4.schedule.survey_days[0]
        scout = replace(self.config.scout, missed_days=(missed_day,))
        result = simulate_scouting(replace(self.config, scout=scout))
        summary = result.surveys[0]
        rows = [
            item for item in result.observations if item.survey_id == summary.survey_id
        ]
        self.assertTrue(summary.missed_survey)
        self.assertEqual(summary.visited_zones, 0)
        self.assertEqual(summary.valid_observations, 0)
        self.assertTrue(all(item.quality_flag == "missed_survey" for item in rows))

    def test_scout_output_contains_no_hidden_truth_fields(self) -> None:
        names = {field.name for field in fields(ScoutObservation)}
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

    def test_invalid_settings_are_rejected(self) -> None:
        values = self.config.scout
        with self.assertRaises(ValueError):
            ScoutSettings((), values.seed, 0.1, 0.8, 0.1, 0.0, 4, 3, ())
        with self.assertRaises(ValueError):
            replace(values, route_zone_ids=("Z_R1_C1", "Z_R1_C1"))
        with self.assertRaises(ValueError):
            replace(values, missing_probability=1.1)

    def test_csv_output_counts_and_overwrite_protection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = replace(self.config.output, directory=Path(directory))
            config = replace(self.config, output=output)
            paths = write_artifacts(config, self.result)
            with paths[0].open(encoding="utf-8", newline="") as stream:
                self.assertEqual(
                    len(tuple(csv.DictReader(stream))), len(self.result.observations)
                )
            with paths[1].open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(tuple(csv.DictReader(stream))), len(self.result.surveys))
            with self.assertRaises(FileExistsError):
                write_artifacts(config, self.result)


if __name__ == "__main__":
    unittest.main()
