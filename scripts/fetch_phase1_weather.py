"""Fetch and convert the one pinned NASA POWER season used by Phase 1."""

from __future__ import annotations

import argparse
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import requests


POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
LATITUDE = 42.03
LONGITUDE = -93.63
START_DATE = date(2022, 1, 1)
END_DATE = date(2022, 12, 31)
POWER_VARIABLES = (
    "TOA_SW_DWN",
    "ALLSKY_SFC_SW_DWN",
    "T2M",
    "T2M_MIN",
    "T2M_MAX",
    "T2MDEW",
    "WS2M",
    "PRECTOTCORR",
    "SNODP",
)


def fetch_power_data() -> dict[str, object]:
    response = requests.get(
        POWER_URL,
        params={
            "parameters": ",".join(POWER_VARIABLES),
            "community": "AG",
            "longitude": LONGITUDE,
            "latitude": LATITUDE,
            "start": START_DATE.strftime("%Y%m%d"),
            "end": END_DATE.strftime("%Y%m%d"),
            "format": "JSON",
            "user": "pcse",
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def convert_to_pcse_csv(power_data: dict[str, object]) -> str:
    header = power_data["header"]
    geometry = power_data["geometry"]
    parameters = power_data["properties"]["parameter"]
    fill_value = float(header["fill_value"])
    elevation = float(geometry["coordinates"][2])

    expected_days = [
        START_DATE + timedelta(days=offset)
        for offset in range((END_DATE - START_DATE).days + 1)
    ]
    ratios: list[float] = []
    rows: list[str] = []
    for day in expected_days:
        key = day.strftime("%Y%m%d")
        values = {
            name: float(parameters[name][key]) for name in POWER_VARIABLES
        }
        missing = [
            name
            for name, value in values.items()
            if value == fill_value or not math.isfinite(value)
        ]
        if missing:
            raise ValueError(f"NASA POWER has missing {missing} on {day}.")

        ratios.append(values["ALLSKY_SFC_SW_DWN"] / values["TOA_SW_DWN"])
        vapour_pressure_kpa = 0.6108 * math.exp(
            (17.27 * values["T2MDEW"]) / (values["T2MDEW"] + 237.3)
        )
        rows.append(
            ",".join(
                (
                    key,
                    f"{values['ALLSKY_SFC_SW_DWN'] * 1000:.1f}",
                    f"{values['T2M_MIN']:.2f}",
                    f"{values['T2M_MAX']:.2f}",
                    f"{vapour_pressure_kpa:.4f}",
                    f"{values['WS2M']:.2f}",
                    f"{values['PRECTOTCORR']:.2f}",
                    f"{values['SNODP']:.2f}",
                )
            )
        )

    angstrom_a = _percentile(ratios, 0.05)
    angstrom_ab = _percentile(ratios, 0.98)
    angstrom_b = angstrom_ab - angstrom_a
    if not (0.1 <= angstrom_a <= 0.4 and 0.3 <= angstrom_b <= 0.7):
        angstrom_a, angstrom_b = 0.29, 0.49

    api_version = header["api"]["version"]
    retrieval_date = date.today().isoformat()
    metadata = [
        "## Site Characteristics",
        "Country = 'United States'",
        "Station = 'Ames, Iowa NASA POWER grid point'",
        "Description = 'Fixed calendar-year 2022 weather for Phase 1, time standard LST'",
        (
            "Source = 'NASA POWER Daily API "
            f"{api_version}, retrieved {retrieval_date}, "
            "radiation MJ/m2/day converted to kJ/m2/day, dewpoint converted "
            "to vapour pressure with FAO-56 equation 14'"
        ),
        "Contact = 'https://power.larc.nasa.gov/'",
        (
            f"Longitude = {LONGITUDE}; Latitude = {LATITUDE}; "
            f"Elevation = {elevation:.2f}; AngstromA = {angstrom_a:.6f}; "
            f"AngstromB = {angstrom_b:.6f}; HasSunshine = False"
        ),
        "## Daily weather observations (missing values are NaN)",
        "DAY,IRRAD,TMIN,TMAX,VAP,WIND,RAIN,SNOWDEPTH",
    ]
    return "\n".join((*metadata, *rows, ""))


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to replace existing file: {args.output}")

    csv_text = convert_to_pcse_csv(fetch_power_data())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(csv_text, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output} ({csv_text.count(chr(10)) - 9} daily rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
