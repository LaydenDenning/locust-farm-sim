"""Run the pinned WOFOST 8.1 maize model for one field zone at a time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from src.farm import FieldZone, Phase1Config, SoilProfile


MODEL_NAME = "Wofost81_NWLP_CWB_CNB"

# PCSE 6.0.13's WOFOST 8.1 module does not publish NNI directly. The required
# NNI column is derived below with the same formula used by PCSE's NPK stress
# module, using the WOFOST 8.1 crop's simulated N amounts and biomass.
DAILY_TRUTH_VARIABLES = (
    "DVS",
    "LAI",
    "TAGP",
    "WLV",
    "WST",
    "WRT",
    "WSO",
    "SM",
    "NAVAIL",
    "NNI",
    "NamountSO",
    "NamountLV",
    "NamountST",
    "NamountRT",
    "NuptakeTotal",
)
ENGINE_OUTPUT_VARIABLES = tuple(
    variable for variable in DAILY_TRUTH_VARIABLES if variable != "NNI"
)

SUMMARY_VARIABLES = (
    "DVS",
    "LAIMAX",
    "TAGP",
    "TWSO",
    "TWLV",
    "TWST",
    "TWRT",
    "CTRAT",
    "RD",
    "DOS",
    "DOE",
    "DOA",
    "DOM",
    "DOH",
    "DOV",
    "CEVST",
    "NuptakeTotal",
    "NamountSO",
)


@dataclass(frozen=True)
class ZoneResult:
    """In-memory daily and terminal truth for a single zone."""

    daily: pd.DataFrame
    summary: dict[str, object]


def _load_pcse_api() -> SimpleNamespace:
    """Import PCSE only when a model is constructed.

    PCSE initializes its user directory on first import. Keeping imports here
    avoids that side effect for callers which only inspect project modules.
    """
    try:
        from pcse.base import ParameterProvider
        from pcse.input import (
            CSVWeatherDataProvider,
            WOFOST81SiteDataProvider_Classic,
            YAMLCropDataProvider,
        )
        from pcse.models import Wofost81_NWLP_CWB_CNB
    except ModuleNotFoundError as exc:
        if exc.name == "pcse":
            raise RuntimeError(
                "PCSE is not installed. Create the py3_pcse environment from "
                "environment.yml before running Phase 1."
            ) from exc
        raise

    return SimpleNamespace(
        ParameterProvider=ParameterProvider,
        CSVWeatherDataProvider=CSVWeatherDataProvider,
        SiteDataProvider=WOFOST81SiteDataProvider_Classic,
        CropDataProvider=YAMLCropDataProvider,
        Model=Wofost81_NWLP_CWB_CNB,
    )


class TruthModel:
    """Reusable local inputs plus isolated WOFOST state for each field zone."""

    def __init__(self, config: Phase1Config) -> None:
        self.config = config
        self._validate_static_inputs()
        self._pcse = _load_pcse_api()

        self._crop_data = self._pcse.CropDataProvider(
            model=self._pcse.Model,
            fpath=str(self.config.crop.parameter_directory),
            force_reload=False,
        )
        self._crop_data.set_active_crop(
            self.config.crop.crop_name,
            self.config.crop.variety_name,
        )
        self._weather = self._pcse.CSVWeatherDataProvider(
            str(self.config.weather.file)
        )

    def run_zone(self, zone: FieldZone, soil_profile: SoilProfile) -> ZoneResult:
        """Run one zone through maturity without writing any output files."""
        planting_date = self.config.calendar.base_sowing_date + timedelta(
            days=zone.planting_offset_days
        )
        self._validate_campaign_dates(planting_date)

        tdwi_kg_ha = zone.tdwi_kg_ha
        soil_data = self._build_soil_data(zone, soil_profile)
        site_data = self._build_site_data(zone, soil_profile)

        # A fresh provider is required because the PCSE engine mutates timer
        # parameters and clears overrides when each crop finishes.
        parameters = self._pcse.ParameterProvider(
            cropdata=self._crop_data,
            soildata=soil_data,
            sitedata=site_data,
        )
        parameters.set_override("TDWI", tdwi_kg_ha)

        engine = self._pcse.Model(
            parameters,
            self._weather,
            self._build_agromanagement(planting_date),
            output_vars=ENGINE_OUTPUT_VARIABLES,
            summary_vars=SUMMARY_VARIABLES,
        )
        engine.run_till_terminate()

        raw_daily = engine.get_output()
        raw_summaries = engine.get_summary_output()
        if not raw_daily:
            raise RuntimeError(f"Zone {zone.zone_id} produced no daily model output.")
        if not raw_summaries:
            raise RuntimeError(f"Zone {zone.zone_id} produced no crop summary.")

        pcse_summary = dict(raw_summaries[-1])
        if self.config.calendar.crop_end_type == "maturity" and not pcse_summary.get(
            "DOM"
        ):
            raise RuntimeError(
                f"Zone {zone.zone_id} did not reach maturity within "
                f"{self.config.calendar.max_duration_days} days."
            )

        metadata = self._zone_metadata(
            zone, soil_profile, planting_date, tdwi_kg_ha, soil_data, site_data
        )
        daily = self._normalize_daily(raw_daily, metadata, zone.zone_id)
        summary = self._normalize_summary(pcse_summary, metadata, planting_date)
        return ZoneResult(daily=daily, summary=summary)

    def _validate_static_inputs(self) -> None:
        if self.config.crop.model_name != MODEL_NAME:
            raise ValueError(
                f"Phase 1 requires model_name={MODEL_NAME!r}; received "
                f"{self.config.crop.model_name!r}."
            )

        parameter_directory = Path(self.config.crop.parameter_directory)
        if not parameter_directory.is_dir():
            raise FileNotFoundError(
                f"Crop parameter directory not found: {parameter_directory}"
            )
        for filename in ("crops.yaml", f"{self.config.crop.crop_name}.yaml"):
            path = parameter_directory / filename
            if not path.is_file():
                raise FileNotFoundError(f"Crop parameter file not found: {path}")

        weather_file = Path(self.config.weather.file)
        if not weather_file.is_file():
            raise FileNotFoundError(f"Weather file not found: {weather_file}")

    def _validate_campaign_dates(self, planting_date: date) -> None:
        campaign_end = planting_date + timedelta(
            days=self.config.calendar.max_duration_days
        )
        if planting_date < self.config.weather.start_date:
            raise ValueError(
                f"Planting date {planting_date} precedes weather coverage "
                f"starting {self.config.weather.start_date}."
            )
        if campaign_end > self.config.weather.end_date:
            raise ValueError(
                f"Campaign through {campaign_end} exceeds weather coverage ending "
                f"{self.config.weather.end_date}."
            )

    def _build_soil_data(
        self, zone: FieldZone, soil_profile: SoilProfile
    ) -> dict[str, float]:
        if zone.slow_drainage:
            sope = self.config.slow_drainage.sope_cm_day
            ksub = self.config.slow_drainage.ksub_cm_day
        else:
            sope = soil_profile.sope_cm_day
            ksub = soil_profile.ksub_cm_day

        return {
            "SMW": float(soil_profile.smw),
            "SMFCF": float(soil_profile.smfcf),
            "SM0": float(soil_profile.sm0),
            "RDMSOL": float(soil_profile.rdmsol_cm),
            "CRAIRC": float(soil_profile.crairc),
            "K0": float(soil_profile.k0_cm_day),
            "SOPE": float(sope),
            "KSUB": float(ksub),
        }

    def _build_site_data(self, zone: FieldZone, soil_profile: SoilProfile) -> Any:
        wav_cm = soil_profile.base_wav_cm
        ssmax_cm = soil_profile.ssmax_cm
        if zone.slow_drainage:
            wav_cm += self.config.slow_drainage.wav_addition_cm
            ssmax_cm = self.config.slow_drainage.ssmax_cm

        return self._pcse.SiteDataProvider(
            WAV=float(wav_cm),
            NAVAILI=float(zone.initial_available_n_kg_ha),
            CO2=float(self.config.site.co2_ppm),
            SSMAX=float(ssmax_cm),
        )

    def _build_agromanagement(self, planting_date: date) -> list[dict[date, object]]:
        crop_calendar = {
            "crop_name": self.config.crop.crop_name,
            "variety_name": self.config.crop.variety_name,
            "crop_start_date": planting_date,
            "crop_start_type": self.config.calendar.crop_start_type,
            "crop_end_date": None,
            "crop_end_type": self.config.calendar.crop_end_type,
            "max_duration": self.config.calendar.max_duration_days,
        }
        return [
            {
                planting_date: {
                    "CropCalendar": crop_calendar,
                    "TimedEvents": None,
                    "StateEvents": None,
                }
            }
        ]

    def _zone_metadata(
        self,
        zone: FieldZone,
        soil_profile: SoilProfile,
        planting_date: date,
        tdwi_kg_ha: float,
        soil_data: dict[str, float],
        site_data: Any,
    ) -> dict[str, object]:
        return {
            "zone_id": zone.zone_id,
            "row": zone.row,
            "column": zone.column,
            "x_m": zone.x_m,
            "y_m": zone.y_m,
            "width_m": zone.width_m,
            "height_m": zone.height_m,
            "soil_profile": zone.soil_profile,
            "planting_date": planting_date,
            "planting_offset_days": zone.planting_offset_days,
            "initial_available_n_kg_ha": zone.initial_available_n_kg_ha,
            "stand_density_plants_m2": zone.stand_density_plants_m2,
            "slow_drainage": zone.slow_drainage,
            "tdwi_kg_ha": tdwi_kg_ha,
            "initial_available_water_cm": site_data["WAV"],
            "soil_smw": soil_profile.smw,
            "soil_smfcf": soil_profile.smfcf,
            "soil_sm0": soil_profile.sm0,
            "soil_rdmsol_cm": soil_data["RDMSOL"],
            "soil_crairc": soil_data["CRAIRC"],
            "soil_k0_cm_day": soil_data["K0"],
            "soil_sope_cm_day": soil_data["SOPE"],
            "soil_ksub_cm_day": soil_data["KSUB"],
            "soil_ssmax_cm": site_data["SSMAX"],
        }

    def _normalize_daily(
        self,
        raw_daily: list[dict[str, object]],
        metadata: dict[str, object],
        zone_id: str,
    ) -> pd.DataFrame:
        daily = pd.DataFrame.from_records(raw_daily).rename(columns={"day": "date"})
        if "date" not in daily:
            raise RuntimeError(f"Zone {zone_id} daily output has no date field.")

        daily["date"] = pd.to_datetime(daily["date"]).dt.date
        if daily["date"].duplicated().any():
            raise RuntimeError(f"Zone {zone_id} produced duplicate daily dates.")

        for variable in ENGINE_OUTPUT_VARIABLES:
            if variable not in daily:
                raise RuntimeError(
                    f"Zone {zone_id} daily output has no {variable} field."
                )
        dvs = pd.to_numeric(daily["DVS"], errors="coerce")
        daily["crop_active"] = dvs.notna() & dvs.ge(0.0)
        daily["NNI"] = daily.apply(self._derive_nni, axis=1)

        active = daily.loc[daily["crop_active"]]
        for variable in DAILY_TRUTH_VARIABLES:
            values = pd.to_numeric(active[variable], errors="coerce")
            if values.isna().any():
                raise RuntimeError(
                    f"Zone {zone_id} has missing active-day {variable} values."
                )

        for name, value in reversed(tuple(metadata.items())):
            daily.insert(0, name, value)

        columns = [
            *metadata,
            "date",
            "crop_active",
            *DAILY_TRUTH_VARIABLES,
        ]
        return daily.loc[:, columns]

    def _derive_nni(self, row: pd.Series) -> float | None:
        """Derive the nitrogen nutrition index with PCSE's standard formula."""

        values = {
            name: pd.to_numeric(row.get(name), errors="coerce")
            for name in ("DVS", "WLV", "WST", "NamountLV", "NamountST")
        }
        if any(pd.isna(value) for value in values.values()) or values["DVS"] < 0:
            return None

        vegetative_biomass = values["WLV"] + values["WST"]
        if vegetative_biomass <= 0:
            return 0.001

        nmax_leaf = _afgen(
            self._crop_data["NMAXLV_TB"], float(values["DVS"])
        )
        nmax_stem = float(self._crop_data["NMAXST_FR"]) * nmax_leaf
        critical_amount = float(self._crop_data["NCRIT_FR"]) * (
            nmax_leaf * values["WLV"] + nmax_stem * values["WST"]
        )
        residual_amount = (
            float(self._crop_data["NRESIDLV"]) * values["WLV"]
            + float(self._crop_data["NRESIDST"]) * values["WST"]
        )
        denominator = critical_amount - residual_amount
        if denominator <= 0:
            return 0.001
        actual_amount = values["NamountLV"] + values["NamountST"]
        return max(
            0.001,
            min(1.0, float((actual_amount - residual_amount) / denominator)),
        )

    @staticmethod
    def _normalize_summary(
        pcse_summary: dict[str, object],
        metadata: dict[str, object],
        planting_date: date,
    ) -> dict[str, object]:
        return {
            **metadata,
            "planting_date": planting_date,
            "emergence_date": pcse_summary.get("DOE"),
            "anthesis_date": pcse_summary.get("DOA"),
            "maturity_date": pcse_summary.get("DOM"),
            **pcse_summary,
        }


def _afgen(table: object, x_value: float) -> float:
    """Linearly interpolate a WOFOST x/y table without adding a dependency."""

    values = [float(value) for value in table]
    if len(values) < 2 or len(values) % 2:
        raise ValueError("WOFOST AFGEN table must contain x/y pairs.")
    points = list(zip(values[0::2], values[1::2], strict=True))
    if x_value <= points[0][0]:
        return points[0][1]
    for (x_left, y_left), (x_right, y_right) in zip(points, points[1:]):
        if x_value <= x_right:
            fraction = (x_value - x_left) / (x_right - x_left)
            return y_left + fraction * (y_right - y_left)
    return points[-1][1]
