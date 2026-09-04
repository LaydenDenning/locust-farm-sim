"""Tests for method-agnostic experiment plots."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.simulation.analyze_experiment import PLOT_FILENAMES, write_experiment_plots


class ExperimentAnalysisTests(unittest.TestCase):
    def test_all_generic_plots_are_written(self) -> None:
        methods = pd.DataFrame({"scenario_id": ["S1", "S1", "S2", "S2"], "method_id": ["drone", "ground", "drone", "ground"], "true_positives": [1, 0, 1, 1], "false_negatives": [0, 1, 0, 0], "false_positives": [0, 0, 1, 0], "unavailable_observations": [0, 2, 1, 2], "avoided_twso_kg": [10, 3, 7, 5], "net_benefit_vs_no_intervention": [4, 2, -1, 1]})
        pairwise = pd.DataFrame({"candidate_method_id": ["drone", "drone"], "reference_method_id": ["ground", "ground"], "net_benefit_delta": [2, -2]})
        sensitivities = pd.DataFrame({"scope": ["comparison", "comparison"], "result_id": ["drone_vs_ground", "drone_vs_ground"], "parameter": ["severity", "cost"], "correlation": [0.5, -0.2]})
        with TemporaryDirectory() as directory:
            paths = write_experiment_plots(methods, pairwise, sensitivities, directory)
            self.assertEqual(tuple(path.name for path in paths), PLOT_FILENAMES)
            self.assertTrue(all(path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for path in paths))


if __name__ == "__main__":
    unittest.main()
