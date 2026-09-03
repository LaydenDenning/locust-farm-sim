"""Integration checks for paired Phase 2 WOFOST stress scenarios."""

from __future__ import annotations

import unittest
from math import isclose
from pathlib import Path

import pandas as pd

from src.farm import load_farm
from src.simulation.run_phase2 import run_phase2
from src.simulation.stress import events_by_zone, load_phase2_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "phase2.yaml"


class Phase2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_phase2_config(CONFIG_PATH)
        cls.farm = load_farm(cls.config.phase1)
        cls.result = run_phase2(cls.config, farm=cls.farm)

    def test_both_scenarios_cover_all_zones_with_unique_keys(self) -> None:
        self.assertEqual(len(self.result.summary), 50)
        self.assertEqual(set(self.result.daily["scenario"]), {"baseline", "stressed"})
        self.assertFalse(
            self.result.daily.duplicated(["scenario", "zone_id", "date"]).any()
        )
        self.assertFalse(
            self.result.summary.duplicated(["scenario", "zone_id"]).any()
        )
        for scenario in ("baseline", "stressed"):
            rows = self.result.summary.loc[
                self.result.summary["scenario"] == scenario
            ]
            self.assertEqual(set(rows["zone_id"]), {z.zone_id for z in self.farm.zones})
            self.assertTrue(rows["maturity_date"].notna().all())

    def test_event_timing_footprint_and_effects_are_measurable(self) -> None:
        configured = {(event.event_id, event.zone_id) for event in self.config.events}
        measured = set(
            zip(self.result.impacts["event_id"], self.result.impacts["zone_id"])
        )
        self.assertEqual(measured, configured)
        self.assertTrue(self.result.impacts["first_lai_divergence_date"].notna().all())
        self.assertTrue((self.result.impacts["yield_loss_kg_ha"] > 0).all())
        self.assertTrue((self.result.impacts["yield_loss_pct"] > 0).all())

    def test_initial_nitrogen_and_stand_treatments_change_only_their_input(self) -> None:
        baseline = self.result.summary.loc[
            self.result.summary["scenario"] == "baseline"
        ].set_index("zone_id")
        stressed = self.result.summary.loc[
            self.result.summary["scenario"] == "stressed"
        ].set_index("zone_id")
        for event in self.config.events:
            if event.stress_type == "nitrogen_deficit":
                self.assertTrue(
                    isclose(
                        stressed.loc[event.zone_id, "initial_available_n_kg_ha"],
                        baseline.loc[event.zone_id, "initial_available_n_kg_ha"]
                        * (1 - event.severity),
                    )
                )
            elif event.stress_type == "stand_loss":
                self.assertTrue(
                    isclose(
                        stressed.loc[event.zone_id, "tdwi_kg_ha"],
                        baseline.loc[event.zone_id, "tdwi_kg_ha"]
                        * (1 - event.severity),
                    )
                )

    def test_water_deficit_caps_active_period_soil_moisture(self) -> None:
        grouped = events_by_zone(self.config.events)
        stressed = self.result.daily.loc[self.result.daily["scenario"] == "stressed"]
        for zone_id, events in grouped.items():
            event = events[0]
            if event.stress_type != "water_deficit":
                continue
            rows = stressed.loc[
                (stressed["zone_id"] == zone_id) & stressed["stress_active"]
            ]
            baseline_rows = self.result.daily.loc[
                (self.result.daily["scenario"] == "baseline")
                & (self.result.daily["zone_id"] == zone_id)
                & self.result.daily["date"].isin(rows["date"])
            ]
            target = rows["soil_smfcf"].iloc[0] - event.severity * (
                rows["soil_smfcf"].iloc[0] - rows["soil_smw"].iloc[0]
            )
            active_sm = pd.to_numeric(rows["SM"], errors="coerce").dropna()
            baseline_sm = pd.to_numeric(
                baseline_rows["SM"], errors="coerce"
            ).dropna()
            self.assertFalse(active_sm.empty)
            self.assertLessEqual(active_sm.mean(), target + 1e-8)
            self.assertTrue((active_sm.to_numpy() < baseline_sm.to_numpy()).all())

    def test_unaffected_zones_match_their_baseline(self) -> None:
        affected = {event.zone_id for event in self.config.events}
        baseline = self.result.summary.loc[
            self.result.summary["scenario"] == "baseline"
        ].set_index("zone_id")
        stressed = self.result.summary.loc[
            self.result.summary["scenario"] == "stressed"
        ].set_index("zone_id")
        for zone_id in set(baseline.index) - affected:
            for variable in ("LAIMAX", "TWSO", "NuptakeTotal"):
                self.assertEqual(
                    baseline.loc[zone_id, variable], stressed.loc[zone_id, variable]
                )

    def test_repeat_run_is_identical(self) -> None:
        repeated = run_phase2(self.config, farm=self.farm)
        pd.testing.assert_frame_equal(self.result.daily, repeated.daily)
        pd.testing.assert_frame_equal(self.result.summary, repeated.summary)
        pd.testing.assert_frame_equal(self.result.impacts, repeated.impacts)


if __name__ == "__main__":
    unittest.main()
