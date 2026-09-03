"""Actual PCSE integration and Phase 1 scientific acceptance checks."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.crop import DAILY_TRUTH_VARIABLES, TruthModel
from src.farm import load_farm, load_phase1_config
from src.simulation.run_phase1 import run_farm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "phase1.yaml"


class Phase1IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # PCSE 6.0.13 emits dependency deprecations and two unclosed-file
        # ResourceWarnings under unittest's stricter warning policy. They are
        # upstream warnings and do not describe failed simulation state.
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", ResourceWarning)
        cls.config = load_phase1_config(CONFIG_PATH)
        cls.farm = load_farm(cls.config)
        cls.model = TruthModel(cls.config)
        cls.result = run_farm(
            cls.config, farm=cls.farm, truth_model=cls.model
        )

    def test_all_zones_reach_maturity_with_complete_active_states(self) -> None:
        self.assertEqual(25, self.result.summary["zone_id"].nunique())
        self.assertTrue(self.result.summary["maturity_date"].notna().all())
        self.assertTrue((~self.result.daily["crop_active"]).any())

        active = self.result.daily.loc[self.result.daily["crop_active"]]
        values = active.loc[:, DAILY_TRUTH_VARIABLES].apply(
            pd.to_numeric, errors="coerce"
        )
        self.assertFalse(values.isna().any().any())
        self.assertTrue((values >= 0).all().all())
        self.assertTrue((values["NNI"] <= 1.0).all())

    def test_configured_farm_has_materially_distinct_trajectories(self) -> None:
        summaries = self.result.summary
        self.assertGreaterEqual(summaries["TWSO"].round(-2).nunique(), 3)
        self.assertGreaterEqual(summaries["LAIMAX"].round(1).nunique(), 3)

        for _, daily in self.result.daily.groupby("zone_id"):
            active_lai = daily.loc[daily["crop_active"], "LAI"]
            self.assertLess(active_lai.diff().abs().max(), 1.0)

    def test_controlled_input_changes_have_expected_effects(self) -> None:
        zone = self.farm.get_zone("Z_R3_C2")
        soil = self.config.soil_profiles["reference"]

        low_n = self.model.run_zone(
            replace(zone, initial_available_n_kg_ha=40.0), soil
        )
        high_n = self.model.run_zone(
            replace(zone, initial_available_n_kg_ha=160.0), soil
        )
        self.assertLess(low_n.summary["NuptakeTotal"], high_n.summary["NuptakeTotal"])
        self.assertLess(low_n.summary["TWSO"], high_n.summary["TWSO"])

        low_water = self.model.run_zone(
            zone, replace(soil, base_wav_cm=6.0)
        )
        reference_water = self.model.run_zone(zone, soil)
        self.assertNotEqual(
            low_water.daily["SM"].tolist(), reference_water.daily["SM"].tolist()
        )
        self.assertTrue(
            low_water.summary["LAIMAX"] != reference_water.summary["LAIMAX"]
            or low_water.summary["TWSO"] != reference_water.summary["TWSO"]
        )

        sparse = self.model.run_zone(
            replace(zone, stand_density_plants_m2=6.0), soil
        )
        dense = self.model.run_zone(
            replace(zone, stand_density_plants_m2=8.0), soil
        )
        comparison_date = sorted(
            set(sparse.daily["date"]) & set(dense.daily["date"])
        )[20]
        sparse_lai = sparse.daily.set_index("date").loc[comparison_date, "LAI"]
        dense_lai = dense.daily.set_index("date").loc[comparison_date, "LAI"]
        self.assertLess(sparse_lai, dense_lai)

        early = self.model.run_zone(
            replace(zone, planting_offset_days=-2), soil
        )
        late = self.model.run_zone(
            replace(zone, planting_offset_days=2), soil
        )
        self.assertLess(early.summary["emergence_date"], late.summary["emergence_date"])
        self.assertLess(early.summary["maturity_date"], late.summary["maturity_date"])

        fast_drainage = self.model.run_zone(
            replace(zone, slow_drainage=False), soil
        )
        slow_drainage = self.model.run_zone(
            replace(zone, slow_drainage=True), soil
        )
        fast_sm = fast_drainage.daily.set_index("date")["SM"]
        slow_sm = slow_drainage.daily.set_index("date")["SM"]
        common_dates = fast_sm.index.intersection(slow_sm.index)
        self.assertGreater(
            (fast_sm.loc[common_dates] - slow_sm.loc[common_dates]).abs().max(),
            0.001,
        )

    def test_complete_farm_run_is_reproducible(self) -> None:
        repeated = run_farm(self.config, farm=self.farm)
        pd.testing.assert_frame_equal(self.result.daily, repeated.daily)
        pd.testing.assert_frame_equal(self.result.summary, repeated.summary)

    def test_csv_round_trip_preserves_keys_counts_and_core_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            daily_path = Path(directory) / "daily.csv"
            summary_path = Path(directory) / "summary.csv"
            self.result.daily.to_csv(daily_path, index=False)
            self.result.summary.to_csv(summary_path, index=False)

            daily = pd.read_csv(daily_path, parse_dates=["date"])
            summary = pd.read_csv(
                summary_path,
                parse_dates=[
                    "planting_date",
                    "emergence_date",
                    "anthesis_date",
                    "maturity_date",
                ],
            )
            self.assertEqual(len(self.result.daily), len(daily))
            self.assertEqual(len(self.result.summary), len(summary))
            self.assertFalse(daily.duplicated(["zone_id", "date"]).any())
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(daily["date"]))
            self.assertTrue(
                pd.api.types.is_numeric_dtype(summary["TWSO"])
            )


if __name__ == "__main__":
    unittest.main()
