"""Unit tests for controlled Phase 2 stress definitions and model hooks."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from src.farm import load_farm, load_phase1_config
from src.simulation.stress import (
    StressConfigError,
    StressEvent,
    events_by_zone,
    load_phase2_config,
    load_stress_events,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE1_CONFIG = PROJECT_ROOT / "config" / "phase1.yaml"
PHASE2_CONFIG = PROJECT_ROOT / "config" / "phase2.yaml"


class StressEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phase1 = load_phase1_config(PHASE1_CONFIG)
        cls.farm = load_farm(cls.phase1)

    def test_phase2_config_loads_explicit_event_footprints(self) -> None:
        config = load_phase2_config(PHASE2_CONFIG)
        self.assertEqual(len(config.events), 6)
        self.assertEqual(
            {event.stress_type for event in config.events},
            {"water_deficit", "nitrogen_deficit", "stand_loss"},
        )
        grouped = events_by_zone(config.events)
        self.assertEqual(set(grouped), {event.zone_id for event in config.events})
        self.assertTrue(config.events_file.is_file())

    def test_dates_are_relative_to_each_zone_planting_date(self) -> None:
        event = StressEvent("WATER", "water_deficit", "Z_R1_C1", 35, 4, 0.5)
        planting = date(2022, 5, 1)
        self.assertEqual(event.start_date(planting), date(2022, 6, 5))
        self.assertEqual(event.end_date(planting), date(2022, 6, 8))
        self.assertTrue(event.is_active(date(2022, 6, 6), planting))
        self.assertFalse(event.is_active(date(2022, 6, 9), planting))

    def test_event_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "stress_type"):
            StressEvent("BAD", "heat", "Z_R1_C1", 0, 1, 0.5)
        with self.assertRaisesRegex(ValueError, "severity"):
            StressEvent("BAD", "water_deficit", "Z_R1_C1", 0, 1, 1.0)
        with self.assertRaisesRegex(ValueError, "start_day"):
            StressEvent("BAD", "water_deficit", "Z_R1_C1", -1, 1, 0.5)

    def test_loader_rejects_unknown_zones(self) -> None:
        text = (
            "event_id,stress_type,zone_id,start_day,duration_days,severity\n"
            "WATER,water_deficit,UNKNOWN,20,5,0.5\n"
        )
        with self.assertRaisesRegex(StressConfigError, "unknown zone_id"):
            self._load_text(text)

    def test_loader_rejects_more_than_one_event_per_zone(self) -> None:
        text = (
            "event_id,stress_type,zone_id,start_day,duration_days,severity\n"
            "WATER,water_deficit,Z_R1_C1,20,5,0.5\n"
            "WATER_2,water_deficit,Z_R1_C1,40,5,0.5\n"
        )
        with self.assertRaisesRegex(StressConfigError, "more than one event"):
            self._load_text(text)

    def test_loader_requires_consistent_footprint_rows(self) -> None:
        text = (
            "event_id,stress_type,zone_id,start_day,duration_days,severity\n"
            "WATER,water_deficit,Z_R1_C1,20,5,0.5\n"
            "WATER,water_deficit,Z_R1_C2,21,5,0.5\n"
        )
        with self.assertRaisesRegex(StressConfigError, "identical event settings"):
            self._load_text(text)

    def test_initial_stresses_must_cover_the_campaign(self) -> None:
        text = (
            "event_id,stress_type,zone_id,start_day,duration_days,severity\n"
            "N,nitrogen_deficit,Z_R1_C1,1,20,0.5\n"
        )
        with self.assertRaisesRegex(StressConfigError, "must start on day 0"):
            self._load_text(text)

    def test_loader_rejects_unexpected_columns(self) -> None:
        text = (
            "event_id,stress_type,zone_id,start_day,duration_days,severity,note\n"
            "WATER,water_deficit,Z_R1_C1,20,5,0.5,x\n"
        )
        with self.assertRaisesRegex(StressConfigError, "columns do not match"):
            self._load_text(text)

    def _load_text(self, text: str) -> tuple[StressEvent, ...]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            path.write_text(text, encoding="utf-8")
            return load_stress_events(
                path,
                farm=self.farm,
                max_duration_days=self.phase1.calendar.max_duration_days,
            )


if __name__ == "__main__":
    unittest.main()
