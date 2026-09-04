"""Tests for modular experiment artifact writing."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.farm import load_farm
from src.simulation.experiment import run_experiment
from src.simulation.profiles import load_experiment_config
from src.simulation.run_experiment import write_artifacts


ROOT = Path(__file__).resolve().parents[1]


class ExperimentWriterTests(unittest.TestCase):
    def test_artifacts_manifest_and_overwrite_protection(self) -> None:
        config = load_experiment_config(ROOT / "config" / "experiments" / "baseline.yaml")
        farm = load_farm(config.phase1)
        truth = pd.DataFrame([
            {"zone_id": zone.zone_id, "date": config.phase1.calendar.base_sowing_date + timedelta(days=day), "crop_active": True, "LAI": 3.0, "NNI": 0.9, "SM": 0.28, "soil_smw": 0.10, "soil_smfcf": 0.30}
            for day in config.methods[0].schedule.days
            for zone in farm.zones
        ])
        baseline = {zone.zone_id: 10_000.0 for zone in farm.zones}
        result = run_experiment(replace(config, scenario_count=2), truth, farm=farm, baseline_twso_kg_ha=baseline)
        with TemporaryDirectory() as directory:
            output = replace(config.output, directory=Path(directory))
            target_config = replace(config, output=output, scenario_count=2)
            paths = write_artifacts(target_config, result)
            self.assertEqual(len(paths), 10)
            self.assertTrue(all(path.exists() for path in paths))
            manifest = json.loads((Path(directory) / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("Synthetic sensitivity evidence", manifest["claim_boundary"])
            self.assertTrue(all(item["sha256"] for item in manifest["inputs"]))
            with self.assertRaises(FileExistsError):
                write_artifacts(target_config, result)


if __name__ == "__main__":
    unittest.main()
