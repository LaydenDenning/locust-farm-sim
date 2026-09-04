"""Tests for Phase 11 analysis plots."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.simulation.analyze_phase11 import PLOT_FILENAMES, analyze_phase11


class Phase11AnalysisTests(unittest.TestCase):
    def _write_inputs(self, directory: Path) -> None:
        pd.DataFrame(
            {
                "mechanism": ["water_deficit", "nitrogen_deficit", "none"],
                "drone_avoided_twso_kg": [500.0, 200.0, 0.0],
                "scout_avoided_twso_kg": [300.0, 250.0, 0.0],
                "drone_net_benefit": [120.0, -20.0, -40.0],
                "scout_net_benefit": [80.0, 10.0, -30.0],
                "drone_advantage_vs_scout": [40.0, -30.0, -10.0],
            }
        ).to_csv(directory / "scenario_results.csv", index=False)
        pd.DataFrame(
            {
                "parameter": ["max_severity", "treatment_cost_multiplier"],
                "correlation_with_drone_advantage": [0.7, -0.4],
            }
        ).to_csv(directory / "sensitivity.csv", index=False)

    def test_analysis_writes_four_png_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_inputs(root)
            paths = analyze_phase11(root)

            self.assertEqual(tuple(path.name for path in paths), PLOT_FILENAMES)
            for path in paths:
                self.assertTrue(path.exists())
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_existing_plots_are_protected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_inputs(root)
            analyze_phase11(root)

            with self.assertRaises(FileExistsError):
                analyze_phase11(root)
            paths = analyze_phase11(root, overwrite=True)
            self.assertEqual(len(paths), 4)

    def test_missing_required_column_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_inputs(root)
            scenarios = pd.read_csv(root / "scenario_results.csv")
            scenarios.drop(columns="drone_net_benefit").to_csv(
                root / "scenario_results.csv", index=False
            )

            with self.assertRaisesRegex(ValueError, "drone_net_benefit"):
                analyze_phase11(root)


if __name__ == "__main__":
    unittest.main()
