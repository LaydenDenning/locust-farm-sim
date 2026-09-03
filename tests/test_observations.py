"""Tests for Phase 4 zone-level synthetic observations."""

from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import fields, replace
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.farm import load_farm
from src.simulation.issues import IssueConfigError, IssueScenario, load_issue_scenarios
from src.simulation.observations import (
    Observation,
    SurveyScheduleConfig,
    load_phase4_config,
    simulate_observations,
)
from src.simulation.run_phase4 import write_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "phase4.yaml"


def _fake_truth(config: object, farm: object) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for campaign_day in config.schedule.survey_days:
        day = config.phase1.calendar.base_sowing_date + timedelta(days=campaign_day)
        for zone in farm.zones:
            records.append(
                {
                    "zone_id": zone.zone_id,
                    "date": day,
                    "crop_active": True,
                    "LAI": 3.0,
                    "NNI": 1.0,
                    "SM": 0.30,
                    "soil_smw": 0.10,
                    "soil_smfcf": 0.30,
                }
            )
    return pd.DataFrame.from_records(records)


class IssueLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase4_config(CONFIG_PATH)
        cls.farm = load_farm(cls.config.phase1)

    def test_phase4_loads_all_generic_issue_mechanisms(self) -> None:
        self.assertEqual(len(self.config.issues), 4)
        self.assertEqual(
            {issue.mechanism for issue in self.config.issues},
            {"water_deficit", "excess_water", "nutrient_deficit", "canopy_damage"},
        )
        self.assertEqual(sum(len(issue.zone_ids) for issue in self.config.issues), 8)

    def test_issue_loader_rejects_unknown_zone(self) -> None:
        text = (
            "issue_id,mechanism,zone_id,onset_day,progression_per_day,"
            "max_severity,visibility_delay_days,visibility_scale,untreated_loss_fraction\n"
            "BAD,canopy_damage,UNKNOWN,20,0.1,0.8,1,0.9,0.2\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "issues.csv"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(IssueConfigError, "unknown zone_id"):
                load_issue_scenarios(
                    path,
                    valid_zone_ids=(zone.zone_id for zone in self.farm.zones),
                    campaign_days=200,
                )


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase4_config(CONFIG_PATH)
        cls.farm = load_farm(cls.config.phase1)
        cls.truth = _fake_truth(cls.config, cls.farm)
        cls.result = simulate_observations(cls.config, cls.truth, farm=cls.farm)

    def test_weekly_schedule_has_unique_zone_records(self) -> None:
        survey_count = len(self.config.schedule.survey_days)
        self.assertEqual(len(self.result.surveys), survey_count)
        self.assertEqual(len(self.result.observations), survey_count * 25)
        keys = {
            (record.survey_id, record.zone_id) for record in self.result.observations
        }
        self.assertEqual(len(keys), len(self.result.observations))

    def test_seeded_observations_are_reproducible(self) -> None:
        repeated = simulate_observations(self.config, self.truth, farm=self.farm)
        self.assertEqual(self.result, repeated)

    def test_channels_indices_and_uncertainty_are_bounded(self) -> None:
        for record in self.result.observations:
            for value in (
                record.relative_red,
                record.relative_green,
                record.relative_blue,
                record.relative_nir,
                record.canopy_cover,
                record.anomaly_score,
                record.uncertainty,
            ):
                if value is not None:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)
            if record.ndvi_like is not None:
                self.assertGreaterEqual(record.ndvi_like, -1.0)
                self.assertLessEqual(record.ndvi_like, 1.0)

    def test_grounded_survey_marks_all_zones_without_sensor_values(self) -> None:
        summary = next(item for item in self.result.surveys if item.campaign_day == 77)
        self.assertTrue(summary.grounded)
        self.assertEqual(summary.covered_zones, 0)
        self.assertEqual(summary.valid_observations, 0)
        rows = [
            item for item in self.result.observations if item.survey_id == summary.survey_id
        ]
        self.assertEqual(len(rows), 25)
        self.assertTrue(all(item.quality_flag == "grounded" for item in rows))
        self.assertTrue(all(item.ndvi_like is None for item in rows))

    def test_surveys_after_a_zone_finishes_use_its_terminal_state(self) -> None:
        final_day = self.config.schedule.survey_days[-1]
        final_date = self.config.phase1.calendar.base_sowing_date + timedelta(
            days=final_day
        )
        shortened = self.truth.loc[
            ~(
                (self.truth["zone_id"] == "Z_R1_C1")
                & (self.truth["date"] == final_date)
            )
        ]
        result = simulate_observations(self.config, shortened, farm=self.farm)
        final = next(
            item
            for item in result.observations
            if item.zone_id == "Z_R1_C1" and item.survey_date == final_date
        )
        self.assertFalse(final.crop_active)

    def test_anomaly_score_uses_only_exported_sensor_fields(self) -> None:
        names = {field.name for field in fields(Observation)}
        self.assertFalse(
            names
            & {
                "issue_id",
                "mechanism",
                "severity",
                "visibility",
                "untreated_loss_fraction",
            }
        )

    def test_visible_canopy_damage_reduces_index_and_increases_anomaly(self) -> None:
        schedule = SurveyScheduleConfig(80, 80, 7, ())
        noise = replace(
            self.config.noise,
            random_noise_std=0.0,
            illumination_std=0.0,
            rgb_bias=0.0,
            nir_bias=0.0,
            spatial_mix_fraction=0.0,
            registration_mix_fraction=0.0,
            cloud_probability=0.0,
            missing_probability=0.0,
        )
        issue = IssueScenario(
            "VISIBLE",
            "canopy_damage",
            ("Z_R3_C3",),
            60,
            0.1,
            0.9,
            0,
            1.0,
            0.3,
        )
        config = replace(
            self.config,
            schedule=schedule,
            noise=noise,
            issues=(issue,),
        )
        truth = _fake_truth(config, self.farm)
        result = simulate_observations(config, truth, farm=self.farm)
        affected = next(item for item in result.observations if item.zone_id == "Z_R3_C3")
        healthy = next(item for item in result.observations if item.zone_id == "Z_R3_C2")
        self.assertLess(affected.ndvi_like, healthy.ndvi_like)
        self.assertLess(affected.canopy_cover, healthy.canopy_cover)
        self.assertGreater(affected.anomaly_score, healthy.anomaly_score)

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
