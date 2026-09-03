"""Unit tests for the Phase 1 PCSE boundary without importing PCSE."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from src.crop.truth_model import ENGINE_OUTPUT_VARIABLES, TruthModel, _afgen


class FakeCropDataProvider(dict[str, object]):
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        super().__init__(
            NMAXLV_TB=[0.0, 0.06, 1.0, 0.02, 2.0, 0.0167],
            NMAXST_FR=0.5,
            NCRIT_FR=1.0,
            NRESIDLV=0.0053,
            NRESIDST=0.0027,
        )
        self.calls.append(kwargs)
        self.active_crop: tuple[str, str] | None = None

    def set_active_crop(self, crop_name: str, variety_name: str) -> None:
        self.active_crop = (crop_name, variety_name)


class FakeWeatherDataProvider:
    calls: list[str] = []

    def __init__(self, filename: str) -> None:
        self.calls.append(filename)


class FakeSiteDataProvider(dict[str, float]):
    calls: list[dict[str, float]] = []

    def __init__(self, **kwargs: float) -> None:
        super().__init__(kwargs)
        self.calls.append(kwargs)


class FakeParameterProvider:
    instances: list["FakeParameterProvider"] = []

    def __init__(self, **kwargs: object) -> None:
        self.sources = kwargs
        self.overrides: dict[str, float] = {}
        self.instances.append(self)

    def set_override(self, name: str, value: float) -> None:
        self.overrides[name] = value


class FakeModel:
    instances: list["FakeModel"] = []
    maturity_date: date | None = date(2022, 9, 20)

    def __init__(
        self,
        parameters: FakeParameterProvider,
        weather: FakeWeatherDataProvider,
        agromanagement: object,
        **kwargs: object,
    ) -> None:
        self.parameters = parameters
        self.weather = weather
        self.agromanagement = agromanagement
        self.kwargs = kwargs
        self.ran = False
        self.instances.append(self)

    def run_till_terminate(self) -> None:
        self.ran = True

    def get_output(self) -> list[dict[str, object]]:
        return [
            {"day": date(2022, 5, 1), "DVS": None, "LAI": None, "TAGP": None},
            {
                "day": date(2022, 5, 2),
                "DVS": 0.0,
                "LAI": 0.02,
                "TAGP": 10.0,
                "WLV": 6.0,
                "WST": 2.0,
                "WRT": 2.0,
                "WSO": 0.0,
                "SM": 0.30,
                "NAVAIL": 95.0,
                "NamountSO": 0.0,
                "NamountLV": 0.30,
                "NamountST": 0.04,
                "NamountRT": 0.06,
                "NuptakeTotal": 1.0,
            },
        ]

    def get_summary_output(self) -> list[dict[str, object]]:
        return [
            {
                "DOE": date(2022, 5, 8),
                "DOA": date(2022, 7, 15),
                "DOM": self.maturity_date,
                "LAIMAX": 4.2,
                "TWSO": 8500.0,
                "TAGP": 17000.0,
                "NuptakeTotal": 145.0,
            }
        ]


def fake_pcse_api() -> SimpleNamespace:
    return SimpleNamespace(
        ParameterProvider=FakeParameterProvider,
        CSVWeatherDataProvider=FakeWeatherDataProvider,
        SiteDataProvider=FakeSiteDataProvider,
        CropDataProvider=FakeCropDataProvider,
        Model=FakeModel,
    )


class TruthModelTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeCropDataProvider.calls.clear()
        FakeWeatherDataProvider.calls.clear()
        FakeSiteDataProvider.calls.clear()
        FakeParameterProvider.instances.clear()
        FakeModel.instances.clear()
        FakeModel.maturity_date = date(2022, 9, 20)

        self.temp_directory = tempfile.TemporaryDirectory()
        root = Path(self.temp_directory.name)
        parameter_directory = root / "crop_parameters"
        parameter_directory.mkdir()
        (parameter_directory / "crops.yaml").touch()
        (parameter_directory / "maize.yaml").touch()
        weather_file = root / "weather.csv"
        weather_file.touch()

        self.config = SimpleNamespace(
            crop=SimpleNamespace(
                parameter_directory=parameter_directory,
                crop_name="maize",
                variety_name="Grain_maize_201",
                model_name="Wofost81_NWLP_CWB_CNB",
            ),
            weather=SimpleNamespace(
                file=weather_file,
                start_date=date(2022, 1, 1),
                end_date=date(2022, 12, 31),
            ),
            calendar=SimpleNamespace(
                base_sowing_date=date(2022, 5, 1),
                crop_start_type="sowing",
                crop_end_type="maturity",
                max_duration_days=200,
            ),
            site=SimpleNamespace(co2_ppm=420.0),
            slow_drainage=SimpleNamespace(
                sope_cm_day=2.0,
                ksub_cm_day=2.0,
                ssmax_cm=3.0,
                wav_addition_cm=4.0,
            ),
        )
        self.zone = SimpleNamespace(
            zone_id="Z_R3_C3",
            row=3,
            column=3,
            x_m=320.0,
            y_m=320.0,
            width_m=160.0,
            height_m=160.0,
            soil_profile="reference",
            planting_offset_days=0,
            initial_available_n_kg_ha=100.0,
            stand_density_plants_m2=8.0,
            slow_drainage=True,
            tdwi_kg_ha=50.0,
        )
        self.soil = SimpleNamespace(
            smw=0.10,
            smfcf=0.30,
            sm0=0.40,
            rdmsol_cm=120.0,
            crairc=0.06,
            k0_cm_day=10.0,
            sope_cm_day=10.0,
            ksub_cm_day=10.0,
            base_wav_cm=16.0,
            ssmax_cm=0.0,
        )

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def make_model(self) -> TruthModel:
        with patch("src.crop.truth_model._load_pcse_api", return_value=fake_pcse_api()):
            return TruthModel(self.config)

    def test_loads_local_inputs_once_without_forcing_crop_cache(self) -> None:
        model = self.make_model()
        model.run_zone(self.zone, self.soil)
        model.run_zone(self.zone, self.soil)

        self.assertEqual(len(FakeCropDataProvider.calls), 1)
        self.assertIs(FakeCropDataProvider.calls[0]["model"], FakeModel)
        self.assertFalse(FakeCropDataProvider.calls[0]["force_reload"])
        self.assertEqual(len(FakeWeatherDataProvider.calls), 1)
        self.assertEqual(len(FakeParameterProvider.instances), 2)
        self.assertIsNot(
            FakeParameterProvider.instances[0], FakeParameterProvider.instances[1]
        )

    def test_run_zone_applies_tdwi_and_slow_drainage_inputs(self) -> None:
        model = self.make_model()
        result = model.run_zone(self.zone, self.soil)

        parameters = FakeParameterProvider.instances[0]
        self.assertEqual(parameters.overrides["TDWI"], 50.0)
        self.assertEqual(parameters.sources["soildata"]["SOPE"], 2.0)
        self.assertEqual(parameters.sources["soildata"]["KSUB"], 2.0)
        self.assertEqual(parameters.sources["soildata"]["K0"], 10.0)
        self.assertEqual(parameters.sources["sitedata"]["WAV"], 20.0)
        self.assertEqual(parameters.sources["sitedata"]["SSMAX"], 3.0)
        self.assertEqual(parameters.sources["sitedata"]["NAVAILI"], 100.0)
        self.assertEqual(parameters.sources["sitedata"]["CO2"], 420.0)
        self.assertTrue(FakeModel.instances[0].ran)
        self.assertEqual(
            FakeModel.instances[0].kwargs["output_vars"], ENGINE_OUTPUT_VARIABLES
        )
        self.assertEqual(result.summary["maturity_date"], date(2022, 9, 20))

    def test_daily_output_has_stable_truth_schema_and_activity_flag(self) -> None:
        result = self.make_model().run_zone(self.zone, self.soil)

        self.assertEqual(result.daily["zone_id"].tolist(), ["Z_R3_C3"] * 2)
        self.assertEqual(result.daily["crop_active"].tolist(), [False, True])
        self.assertIn("NNI", result.daily.columns)
        self.assertTrue(result.daily["NNI"].isna().iloc[0])
        self.assertAlmostEqual(0.791014, result.daily["NNI"].iloc[1], places=5)
        self.assertEqual(result.daily["date"].tolist()[0], date(2022, 5, 1))

    def test_model_requires_maturity_when_configured(self) -> None:
        FakeModel.maturity_date = None
        with self.assertRaisesRegex(RuntimeError, "did not reach maturity"):
            self.make_model().run_zone(self.zone, self.soil)

    def test_rejects_campaign_outside_weather_coverage(self) -> None:
        self.config.calendar.max_duration_days = 300
        with self.assertRaisesRegex(ValueError, "exceeds weather coverage"):
            self.make_model().run_zone(self.zone, self.soil)

    def test_rejects_a_different_pcse_model(self) -> None:
        self.config.crop.model_name = "Wofost81_PP"
        with patch("src.crop.truth_model._load_pcse_api") as loader:
            with self.assertRaisesRegex(ValueError, "requires model_name"):
                TruthModel(self.config)
        loader.assert_not_called()

    def test_local_temperate_maize_has_required_wofost81_compatibility(self) -> None:
        parameter_file = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "crop_parameters"
            / "wofost81"
            / "maize.yaml"
        )
        raw = yaml.safe_load(parameter_file.read_text(encoding="utf-8"))
        temperate = raw["CropParameters"]["EcoTypes"]["temperate_maize"]
        required = {
            "AMAX_REF",
            "AMAX_SLP",
            "AMAX_LNB",
            "KN",
            "DVS_N_TRANSL",
            "NSLLV_TB",
            "RGRLAI_MIN",
            "REALLOC_DVS",
            "REALLOC_STEM_FRACTION",
            "REALLOC_LEAF_FRACTION",
            "REALLOC_STEM_RATE",
            "REALLOC_LEAF_RATE",
            "REALLOC_EFFICIENCY",
        }
        self.assertTrue(required.issubset(temperate))
        self.assertGreater(temperate["REALLOC_DVS"][0], 2.0)

    def test_afgen_uses_endpoints_outside_the_table_range(self) -> None:
        table = [0.0, 10.0, 1.0, 20.0]
        self.assertEqual(10.0, _afgen(table, -1.0))
        self.assertEqual(15.0, _afgen(table, 0.5))
        self.assertEqual(20.0, _afgen(table, 2.0))


if __name__ == "__main__":
    unittest.main()
