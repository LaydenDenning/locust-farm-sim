"""Tests for deterministic Phase 1 field and configuration inputs."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import date
from pathlib import Path

from src.farm import (
    ConfigError,
    Farm,
    SoilProfile,
    load_farm,
    load_phase1_config,
    load_zones_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "phase1.yaml"


class Phase1ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase1_config(CONFIG_PATH)
        cls.farm = load_farm(cls.config)

    def test_global_phase1_values(self) -> None:
        self.assertEqual((800.0, 800.0), (self.farm.width_m, self.farm.height_m))
        self.assertEqual((5, 5), (self.farm.rows, self.farm.columns))
        self.assertEqual("maize", self.config.crop.crop_name)
        self.assertEqual("Grain_maize_201", self.config.crop.variety_name)
        self.assertEqual("Wofost81_NWLP_CWB_CNB", self.config.crop.model_name)
        self.assertEqual(date(2022, 5, 1), self.config.calendar.base_sowing_date)
        self.assertEqual(200, self.config.calendar.max_duration_days)
        self.assertEqual(420.0, self.config.site.co2_ppm)
        self.assertEqual(date(2022, 1, 1), self.config.weather.start_date)
        self.assertEqual(date(2022, 12, 31), self.config.weather.end_date)

    def test_paths_are_resolved_from_project_root(self) -> None:
        self.assertEqual(PROJECT_ROOT, self.config.project_root)
        self.assertEqual(
            PROJECT_ROOT / "data" / "phase1" / "zones.csv",
            self.config.zones_file,
        )
        self.assertEqual(
            PROJECT_ROOT / "data" / "weather" / "ames_2022.csv",
            self.config.weather.file,
        )
        self.assertEqual(PROJECT_ROOT / "outputs" / "phase1", self.config.output.directory)

    def test_soil_profiles_match_approved_synthetic_values(self) -> None:
        expected = {
            "low": (0.07, 0.22, 0.38, 100.0, 0.08, 15.0, 15.0, 15.0, 10.0),
            "reference": (
                0.10,
                0.30,
                0.40,
                120.0,
                0.06,
                10.0,
                10.0,
                10.0,
                16.0,
            ),
            "high": (0.16, 0.36, 0.48, 120.0, 0.06, 5.0, 4.0, 4.0, 22.0),
        }
        for name, values in expected.items():
            profile = self.config.soil_profiles[name]
            self.assertEqual(
                values,
                (
                    profile.smw,
                    profile.smfcf,
                    profile.sm0,
                    profile.rdmsol_cm,
                    profile.crairc,
                    profile.k0_cm_day,
                    profile.sope_cm_day,
                    profile.ksub_cm_day,
                    profile.base_wav_cm,
                ),
            )
            self.assertLess(profile.smw, profile.smfcf)
            self.assertLess(profile.smfcf, profile.sm0)

    def test_slow_drainage_overrides_match_approved_values(self) -> None:
        drainage = self.config.slow_drainage
        self.assertEqual(2.0, drainage.sope_cm_day)
        self.assertEqual(2.0, drainage.ksub_cm_day)
        self.assertEqual(3.0, drainage.ssmax_cm)
        self.assertEqual(4.0, drainage.wav_addition_cm)

    def test_weather_must_cover_all_possible_crop_days(self) -> None:
        short_weather = replace(self.config.weather, end_date=date(2022, 6, 1))
        with self.assertRaisesRegex(ConfigError, "coverage ends"):
            load_farm(replace(self.config, weather=short_weather))

    def test_phase1_calendar_modes_are_fixed(self) -> None:
        with self.assertRaisesRegex(ValueError, "crop_start_type"):
            replace(self.config.calendar, crop_start_type="emergence")
        with self.assertRaisesRegex(ValueError, "crop_end_type"):
            replace(self.config.calendar, crop_end_type="harvest")


class FarmLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase1_config(CONFIG_PATH)
        cls.farm = load_farm(cls.config)

    def test_grid_has_25_unique_complete_zones(self) -> None:
        self.assertEqual(25, len(self.farm.zones))
        self.assertEqual(25, len({zone.zone_id for zone in self.farm.zones}))
        self.assertEqual(
            {(row, column) for row in range(1, 6) for column in range(1, 6)},
            {(zone.row, zone.column) for zone in self.farm.zones},
        )
        self.assertEqual(
            self.farm.width_m * self.farm.height_m,
            sum(zone.area_m2 for zone in self.farm.zones),
        )
        self.assertTrue(
            all(zone.width_m == 160 and zone.height_m == 160 for zone in self.farm.zones)
        )

    def test_zone_ids_and_coordinates_follow_the_explicit_grid(self) -> None:
        for zone in self.farm.zones:
            self.assertEqual(f"Z_R{zone.row}_C{zone.column}", zone.zone_id)
            self.assertEqual((zone.column - 1) * 160.0, zone.x_m)
            self.assertEqual((zone.row - 1) * 160.0, zone.y_m)

    def test_row_patterns(self) -> None:
        soil_by_row = {1: "low", 2: "low", 3: "reference", 4: "high", 5: "high"}
        offset_by_row = {1: -2, 2: -1, 3: 0, 4: 1, 5: 2}
        for zone in self.farm.zones:
            self.assertEqual(soil_by_row[zone.row], zone.soil_profile)
            self.assertEqual(offset_by_row[zone.row], zone.planting_offset_days)

    def test_column_nitrogen_gradient(self) -> None:
        n_by_column = {1: 40.0, 2: 70.0, 3: 100.0, 4: 130.0, 5: 160.0}
        for zone in self.farm.zones:
            self.assertEqual(
                n_by_column[zone.column], zone.initial_available_n_kg_ha
            )

    def test_boundary_density_pattern_and_tdwi(self) -> None:
        for zone in self.farm.zones:
            on_vertical_edge = zone.column in {1, 5}
            on_horizontal_edge = zone.row in {1, 5}
            if on_vertical_edge and on_horizontal_edge:
                expected_density = 6.0
            elif on_vertical_edge or on_horizontal_edge:
                expected_density = 7.0
            else:
                expected_density = 8.0
            self.assertEqual(expected_density, zone.stand_density_plants_m2)
            self.assertEqual(50.0 * expected_density / 8.0, zone.tdwi_kg_ha)

    def test_central_four_zones_are_the_only_slow_drainage_zones(self) -> None:
        actual = {zone.zone_id for zone in self.farm.zones if zone.slow_drainage}
        self.assertEqual(
            {"Z_R3_C3", "Z_R3_C4", "Z_R4_C3", "Z_R4_C4"}, actual
        )

    def test_models_are_immutable(self) -> None:
        zone = self.farm.get_zone("Z_R1_C1")
        with self.assertRaises(FrozenInstanceError):
            zone.row = 2  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            self.farm.rows = 4  # type: ignore[misc]
        with self.assertRaises(TypeError):
            self.config.soil_profiles["new"] = self.config.soil_profiles["low"]

    def test_overlap_is_rejected(self) -> None:
        zones = list(self.farm.zones)
        zones[1] = replace(zones[1], x_m=0.0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            Farm(800.0, 800.0, 5, 5, tuple(zones))

    def test_incomplete_coverage_is_rejected(self) -> None:
        zones = list(self.farm.zones)
        zones[-1] = replace(zones[-1], width_m=159.0)
        with self.assertRaisesRegex(ValueError, "complete field"):
            Farm(800.0, 800.0, 5, 5, tuple(zones))


class InvalidInputTests(unittest.TestCase):
    def test_invalid_soil_water_order_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "SMW < SMFCF < SM0"):
            SoilProfile(
                name="invalid",
                smw=0.30,
                smfcf=0.20,
                sm0=0.40,
                rdmsol_cm=100,
                crairc=0.08,
                k0_cm_day=10,
                sope_cm_day=10,
                ksub_cm_day=10,
                base_wav_cm=10,
            )

    def test_negative_zone_nitrogen_is_rejected(self) -> None:
        header = (
            "zone_id,row,column,x_m,y_m,width_m,height_m,soil_profile,"
            "planting_offset_days,initial_available_n_kg_ha,"
            "stand_density_plants_m2,slow_drainage\n"
        )
        row = "Z_BAD,1,1,0,0,160,160,low,0,-1,8,false\n"
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "zones.csv"
            csv_path.write_text(header + row, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "nonnegative"):
                load_zones_csv(csv_path)

    def test_unexpected_csv_column_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "zones.csv"
            csv_path.write_text("zone_id,unexpected\nZ_1,value\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "columns do not match"):
                load_zones_csv(csv_path)


if __name__ == "__main__":
    unittest.main()
