"""Tests for modular experiment profile loading."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.simulation.profiles import ExperimentConfigError, ScheduleConfig, load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


class ProfileTests(unittest.TestCase):
    def test_baseline_profiles_resolve_and_are_independent(self) -> None:
        config = load_experiment_config(ROOT / "config" / "experiments" / "baseline.yaml")
        self.assertEqual(config.crop_profile.crop.crop_name, "maize")
        self.assertEqual(config.reference_method_id, "ground_weekly")
        self.assertEqual([item.profile.kind for item in config.methods], ["drone", "ground_scout"])
        self.assertEqual(config.methods[0].schedule.days, tuple(range(35, 99, 7)))

    def test_frequency_variants_have_unique_ids_and_schedules(self) -> None:
        config = load_experiment_config(ROOT / "config" / "experiments" / "drone_frequency.yaml")
        self.assertEqual(len(config.methods), 4)
        self.assertEqual(len({item.method_id for item in config.methods}), 4)
        self.assertEqual(len(config.methods[0].schedule.days), 22)

    def test_schedule_rejects_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            ScheduleConfig((7, 7))

    def test_impossible_camera_rate_is_rejected(self) -> None:
        source = (ROOT / "config" / "profiles" / "methods" / "drone_mavic4_rgb_nir.yaml").read_text(encoding="utf-8")
        source = source.replace("minimum_capture_interval_seconds: 1.0", "minimum_capture_interval_seconds: 100.0")
        from src.simulation.profiles import load_method_profile
        with TemporaryDirectory() as directory:
            path = Path(directory) / "method.yaml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaisesRegex(ExperimentConfigError, "captures faster"):
                load_method_profile(path)

    def test_reference_must_exist(self) -> None:
        source = (ROOT / "config" / "experiments" / "baseline.yaml").read_text(encoding="utf-8")
        source = source.replace("farm_config: ../phase1.yaml", f"farm_config: {ROOT / 'config' / 'phase1.yaml'}")
        source = source.replace("../profiles/crops/", f"{ROOT / 'config' / 'profiles' / 'crops'}/")
        source = source.replace("../profiles/scenarios/", f"{ROOT / 'config' / 'profiles' / 'scenarios'}/")
        source = source.replace("../profiles/interventions/", f"{ROOT / 'config' / 'profiles' / 'interventions'}/")
        source = source.replace("../profiles/economics/", f"{ROOT / 'config' / 'profiles' / 'economics'}/")
        source = source.replace("../profiles/methods/", f"{ROOT / 'config' / 'profiles' / 'methods'}/")
        source = source.replace("../phase11.yaml", f"{ROOT / 'config' / 'phase11.yaml'}")
        source = source.replace("reference_method_id: ground_weekly", "reference_method_id: missing")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.yaml"
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(ExperimentConfigError):
                load_experiment_config(path)


if __name__ == "__main__":
    unittest.main()
