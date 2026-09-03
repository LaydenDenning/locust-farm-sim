"""Tests for deterministic Phase 3 drone mission planning."""

from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from math import isclose
from pathlib import Path

from src.farm import load_farm
from src.simulation.drone import (
    CameraConfig,
    load_phase3_config,
    plan_mission,
)
from src.simulation.run_phase3 import write_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "phase3.yaml"


class DroneConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase3_config(CONFIG_PATH)

    def test_manufacturer_and_provisional_settings_load(self) -> None:
        self.assertEqual(self.config.drone.aircraft_name, "DJI Mavic 4 Pro proxy")
        self.assertEqual(self.config.drone.reference_mass_g, 1063)
        self.assertEqual(self.config.drone.maximum_flight_time_minutes, 51)
        self.assertEqual(self.config.drone.maximum_speed_m_s, 25)
        self.assertEqual(self.config.drone.mapping_speed_m_s, 9)
        self.assertAlmostEqual(
            self.config.drone.usable_endurance_seconds,
            51 * 60 * 0.75 * 0.85,
        )

    def test_exact_rgb_and_nir_camera_models_load(self) -> None:
        cameras = {camera.channel: camera for camera in self.config.cameras}
        self.assertEqual((cameras["rgb"].width_px, cameras["rgb"].height_px), (9248, 6944))
        self.assertEqual(cameras["rgb"].horizontal_fov_deg, 68)
        self.assertEqual((cameras["nir"].width_px, cameras["nir"].height_px), (4056, 3040))
        self.assertEqual(cameras["nir"].horizontal_fov_deg, 65)

    def test_camera_rejects_invalid_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "horizontal_fov_deg"):
            CameraConfig("bad", "rgb", 100, 100, 180, 50, 1)


class MissionPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase3_config(CONFIG_PATH)
        cls.farm = load_farm(cls.config.phase1)
        cls.plan = plan_mission(cls.config, farm=cls.farm)

    def test_narrower_camera_controls_shared_footprint(self) -> None:
        nir = next(camera for camera in self.config.cameras if camera.channel == "nir")
        self.assertTrue(
            isclose(
                self.plan.footprint_width_m,
                nir.ground_width_m(self.config.drone.altitude_m),
            )
        )
        self.assertTrue(
            isclose(
                self.plan.footprint_length_m,
                nir.ground_length_m(self.config.drone.altitude_m),
            )
        )

    def test_route_covers_every_zone_without_duplicate_keys(self) -> None:
        self.assertEqual(len(self.plan.coverage), 25)
        self.assertEqual(
            {item.zone_id for item in self.plan.coverage},
            {zone.zone_id for zone in self.farm.zones},
        )
        self.assertTrue(all(item.covered for item in self.plan.coverage))
        self.assertTrue(
            all(isclose(item.coverage_fraction, 1.0) for item in self.plan.coverage)
        )
        self.assertTrue(
            all(item.first_observation_time is not None for item in self.plan.coverage)
        )

    def test_line_spacing_and_capture_timing_are_consistent(self) -> None:
        maximum_spacing = self.plan.footprint_width_m * (
            1 - self.config.drone.side_overlap
        )
        self.assertLessEqual(self.plan.line_spacing_m, maximum_spacing + 1e-9)
        self.assertAlmostEqual(
            self.plan.capture_interval_seconds,
            self.plan.capture_spacing_m / self.config.drone.mapping_speed_m_s,
        )
        for line in self.plan.lines:
            self.assertAlmostEqual(line.distance_m, self.farm.width_m)
            self.assertAlmostEqual(
                line.end_seconds - line.start_seconds,
                self.farm.width_m / self.config.drone.mapping_speed_m_s,
            )

    def test_route_splits_and_every_sortie_respects_usable_endurance(self) -> None:
        self.assertGreater(len(self.plan.sorties), 1)
        self.assertEqual(
            {line.sortie_id for line in self.plan.lines},
            {sortie.sortie_id for sortie in self.plan.sorties},
        )
        for sortie in self.plan.sorties:
            self.assertLessEqual(
                sortie.flight_time_seconds, sortie.usable_endurance_seconds
            )
            self.assertGreaterEqual(sortie.battery_margin_seconds, 0)

    def test_counts_are_internally_consistent(self) -> None:
        self.assertEqual(
            sum(sortie.line_count for sortie in self.plan.sorties),
            len(self.plan.lines),
        )
        for sortie in self.plan.sorties:
            lines = [line for line in self.plan.lines if line.sortie_id == sortie.sortie_id]
            captures = sum(line.capture_count_per_camera for line in lines)
            self.assertEqual(sortie.capture_count_per_camera, captures)
            self.assertEqual(sortie.total_image_count, captures * 2)

    def test_repeat_plan_is_identical(self) -> None:
        self.assertEqual(self.plan, plan_mission(self.config, farm=self.farm))

    def test_grounded_mission_records_every_zone_as_missed(self) -> None:
        plan = plan_mission(replace(self.config, grounded=True), farm=self.farm)
        self.assertEqual(plan.lines, ())
        self.assertEqual(plan.sorties, ())
        self.assertEqual(len(plan.coverage), 25)
        self.assertTrue(all(not item.covered for item in plan.coverage))
        self.assertTrue(all(item.coverage_fraction == 0 for item in plan.coverage))

    def test_impossible_capture_rate_is_rejected(self) -> None:
        cameras = tuple(
            replace(camera, minimum_capture_interval_seconds=10)
            for camera in self.config.cameras
        )
        with self.assertRaisesRegex(ValueError, "captures faster"):
            plan_mission(replace(self.config, cameras=cameras), farm=self.farm)

    def test_csv_outputs_have_expected_rows_and_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = replace(self.config.output, directory=Path(directory))
            config = replace(self.config, output=output)
            paths = write_artifacts(config, self.plan)
            self.assertEqual(len(paths), 3)
            with paths[0].open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(tuple(csv.DictReader(stream))), len(self.plan.lines))
            with paths[1].open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(tuple(csv.DictReader(stream))), len(self.plan.sorties))
            with paths[2].open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(tuple(csv.DictReader(stream))), 25)
            with self.assertRaises(FileExistsError):
                write_artifacts(config, self.plan)


if __name__ == "__main__":
    unittest.main()
