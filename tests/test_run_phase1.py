"""Tests for Phase 1 orchestration without invoking the PCSE engine."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from src.crop import DAILY_TRUTH_VARIABLES, ZoneResult
from src.farm import load_farm, load_phase1_config
from src.simulation.run_phase1 import (
    FarmResult,
    run_farm,
    validate_weather_file,
    write_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "phase1.yaml"


class FakeTruthModel:
    def __init__(self) -> None:
        self.zone_ids: list[str] = []

    def run_zone(self, zone: object, soil: object) -> ZoneResult:
        self.zone_ids.append(zone.zone_id)
        planting_date = date(2022, 5, 1)
        metadata = {
            "zone_id": zone.zone_id,
            "row": zone.row,
            "column": zone.column,
            "x_m": zone.x_m,
            "y_m": zone.y_m,
            "width_m": zone.width_m,
            "height_m": zone.height_m,
            "soil_profile": zone.soil_profile,
            "planting_date": planting_date,
            "planting_offset_days": zone.planting_offset_days,
            "initial_available_n_kg_ha": zone.initial_available_n_kg_ha,
            "stand_density_plants_m2": zone.stand_density_plants_m2,
            "slow_drainage": zone.slow_drainage,
            "tdwi_kg_ha": zone.tdwi_kg_ha,
            "initial_available_water_cm": soil.base_wav_cm,
            "soil_smw": soil.smw,
            "soil_smfcf": soil.smfcf,
            "soil_sm0": soil.sm0,
        }
        values = {
            "DVS": 2.0,
            "LAI": 1.0,
            "TAGP": 10.0,
            "WLV": 2.0,
            "WST": 2.0,
            "WRT": 2.0,
            "WSO": 4.0,
            "SM": soil.smfcf,
            "NAVAIL": 1.0,
            "NNI": 0.8,
            "NamountSO": 1.0,
            "NamountLV": 1.0,
            "NamountST": 1.0,
            "NamountRT": 1.0,
            "NuptakeTotal": 4.0,
        }
        daily = pd.DataFrame.from_records(
            [{**metadata, "date": date(2022, 8, 1), "crop_active": True, **values}]
        )
        summary = {
            **metadata,
            "emergence_date": date(2022, 5, 10),
            "anthesis_date": date(2022, 7, 1),
            "maturity_date": date(2022, 8, 1),
            "LAIMAX": 1.0,
            "TWSO": 4.0,
            "TAGP": 10.0,
            "NuptakeTotal": 4.0,
            "TWLV": 2.0,
            "TWST": 2.0,
            "TWRT": 2.0,
        }
        return ZoneResult(daily=daily, summary=summary)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase1_config(CONFIG_PATH)
        cls.farm = load_farm(cls.config)

    def test_run_farm_collects_every_zone_in_grid_order(self) -> None:
        model = FakeTruthModel()
        result = run_farm(self.config, farm=self.farm, truth_model=model)

        self.assertEqual(25, len(model.zone_ids))
        self.assertEqual("Z_R1_C1", model.zone_ids[0])
        self.assertEqual("Z_R5_C5", model.zone_ids[-1])
        self.assertEqual(25, len(result.daily))
        self.assertEqual(25, len(result.summary))
        self.assertFalse(result.daily.duplicated(["zone_id", "date"]).any())
        self.assertTrue((result.summary["total_biomass_kg_ha"] == 10.0).all())
        self.assertTrue(set(DAILY_TRUTH_VARIABLES).issubset(result.daily.columns))

    def test_same_fake_inputs_produce_identical_sorted_results(self) -> None:
        first = run_farm(
            self.config, farm=self.farm, truth_model=FakeTruthModel()
        )
        second = run_farm(
            self.config, farm=self.farm, truth_model=FakeTruthModel()
        )
        pd.testing.assert_frame_equal(first.daily, second.daily)
        pd.testing.assert_frame_equal(first.summary, second.summary)

    def test_refuses_to_replace_an_existing_artifact(self) -> None:
        result = run_farm(
            self.config, farm=self.farm, truth_model=FakeTruthModel()
        )
        with tempfile.TemporaryDirectory() as directory:
            output = replace(self.config.output, directory=Path(directory))
            config = replace(self.config, output=output)
            (Path(directory) / output.daily_truth_filename).touch()
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                write_artifacts(config, result)


class WeatherValidationTests(unittest.TestCase):
    def test_complete_numeric_weather_is_accepted(self) -> None:
        config = load_phase1_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            weather_path = Path(directory) / "weather.csv"
            weather_path.write_text(_two_day_weather(), encoding="utf-8")
            weather = replace(
                config.weather,
                file=weather_path,
                start_date=date(2022, 1, 1),
                end_date=date(2022, 1, 2),
            )
            validate_weather_file(replace(config, weather=weather))

    def test_missing_weather_value_is_rejected(self) -> None:
        config = load_phase1_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            weather_path = Path(directory) / "weather.csv"
            weather_path.write_text(
                _two_day_weather().replace("0.10,2.00", "NaN,2.00", 1),
                encoding="utf-8",
            )
            weather = replace(
                config.weather,
                file=weather_path,
                start_date=date(2022, 1, 1),
                end_date=date(2022, 1, 2),
            )
            with self.assertRaisesRegex(ValueError, "missing/nonfinite"):
                validate_weather_file(replace(config, weather=weather))


def _two_day_weather() -> str:
    return """## Site Characteristics
Country = 'United States'
Station = 'Test'
Description = 'Test data'
Source = 'Test'
Contact = 'Test'
Longitude = -93.63; Latitude = 42.03; Elevation = 300; AngstromA = 0.2; AngstromB = 0.5; HasSunshine = False
## Daily weather observations (missing values are NaN)
DAY,IRRAD,TMIN,TMAX,VAP,WIND,RAIN,SNOWDEPTH
20220101,5000,-5,2,0.10,2.00,0.5,1.0
20220102,5100,-4,3,0.12,2.10,0.0,1.0
"""


if __name__ == "__main__":
    unittest.main()
