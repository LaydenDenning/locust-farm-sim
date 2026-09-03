"""Command-line runner for the deterministic Phase 1 ground-truth farm."""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from math import isclose, isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.crop import DAILY_TRUTH_VARIABLES, TruthModel
from src.farm import Farm, Phase1Config, load_farm, load_phase1_config


WEATHER_COLUMNS = (
    "DAY",
    "IRRAD",
    "TMIN",
    "TMAX",
    "VAP",
    "WIND",
    "RAIN",
    "SNOWDEPTH",
)


@dataclass(frozen=True)
class FarmResult:
    """Combined in-memory outputs from every zone in one farm run."""

    daily: pd.DataFrame
    summary: pd.DataFrame


def validate_weather_file(config: Phase1Config) -> None:
    """Validate the fixed PCSE CSV before any model engine is started."""

    weather_path = config.weather.file
    try:
        lines = weather_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ValueError(f"Unable to read weather file {weather_path}: {exc}") from exc

    marker = "## Daily weather observations"
    try:
        marker_index = next(
            index for index, line in enumerate(lines) if line.startswith(marker)
        )
    except StopIteration as exc:
        raise ValueError(
            f"Weather file {weather_path} has no daily-observations marker."
        ) from exc

    metadata = _parse_weather_metadata(lines[1:marker_index], weather_path)
    for key, expected in (
        ("Latitude", config.weather.latitude),
        ("Longitude", config.weather.longitude),
    ):
        try:
            actual = float(metadata[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Weather metadata {key} is missing or invalid.") from exc
        if not isclose(actual, expected, abs_tol=1e-6):
            raise ValueError(
                f"Weather metadata {key}={actual} does not match config value "
                f"{expected}."
            )

    reader = csv.DictReader(lines[marker_index + 1 :])
    if reader.fieldnames is None:
        raise ValueError(f"Weather file {weather_path} has no daily header.")
    missing_columns = set(WEATHER_COLUMNS) - set(reader.fieldnames)
    if missing_columns:
        raise ValueError(
            f"Weather file is missing columns: {sorted(missing_columns)}"
        )

    days: set[date] = set()
    for line_number, row in enumerate(reader, start=marker_index + 3):
        try:
            day = date.fromisoformat(
                f"{row['DAY'][0:4]}-{row['DAY'][4:6]}-{row['DAY'][6:8]}"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Weather line {line_number} has an invalid YYYYMMDD date."
            ) from exc
        if day in days:
            raise ValueError(f"Weather file contains duplicate day {day}.")
        days.add(day)

        values: dict[str, float] = {}
        for name in WEATHER_COLUMNS[1:]:
            try:
                value = float(row[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Weather line {line_number} has invalid {name}."
                ) from exc
            if not isfinite(value):
                raise ValueError(
                    f"Weather line {line_number} has a missing/nonfinite {name}."
                )
            values[name] = value

        if values["TMIN"] > values["TMAX"]:
            raise ValueError(f"Weather line {line_number} has TMIN above TMAX.")
        for name in ("IRRAD", "WIND", "RAIN", "SNOWDEPTH"):
            if values[name] < 0:
                raise ValueError(
                    f"Weather line {line_number} has negative {name}."
                )

    expected_days = _date_set(
        config.weather.start_date, config.weather.end_date
    )
    missing_days = sorted(expected_days - days)
    extra_days = sorted(days - expected_days)
    if missing_days or extra_days:
        details = []
        if missing_days:
            details.append(
                f"missing {len(missing_days)} day(s), first {missing_days[0]}"
            )
        if extra_days:
            details.append(
                f"has {len(extra_days)} out-of-range day(s), first {extra_days[0]}"
            )
        raise ValueError("Weather coverage mismatch: " + "; ".join(details))


def run_farm(
    config: Phase1Config,
    *,
    farm: Farm | None = None,
    truth_model: TruthModel | None = None,
) -> FarmResult:
    """Run all zones sequentially and return results without writing files."""

    farm = farm or load_farm(config)
    truth_model = truth_model or TruthModel(config)
    daily_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for zone in sorted(farm.zones, key=lambda item: (item.row, item.column)):
        soil_profile = config.soil_profiles[zone.soil_profile]
        result = truth_model.run_zone(zone, soil_profile)
        daily_frames.append(result.daily)
        summaries.append(result.summary)

    if not daily_frames:
        raise RuntimeError("The farm contains no zone results.")

    daily = pd.concat(daily_frames, ignore_index=True)
    summary = pd.DataFrame.from_records(summaries)
    summary["total_biomass_kg_ha"] = summary[
        ["TWLV", "TWST", "TWRT", "TWSO"]
    ].sum(axis=1, min_count=4)
    daily = daily.sort_values(["row", "column", "date"]).reset_index(drop=True)
    summary = summary.sort_values(["row", "column"]).reset_index(drop=True)

    combined = FarmResult(daily=daily, summary=summary)
    validate_farm_result(combined, farm)
    return combined


def validate_farm_result(result: FarmResult, farm: Farm) -> None:
    """Check model-output invariants required for a usable Phase 1 result."""

    daily = result.daily
    summary = result.summary
    expected_zone_ids = {zone.zone_id for zone in farm.zones}

    required_daily = {
        "zone_id",
        "date",
        "crop_active",
        "soil_smw",
        "soil_sm0",
        *DAILY_TRUTH_VARIABLES,
    }
    missing_daily = required_daily - set(daily.columns)
    if missing_daily:
        raise RuntimeError(
            f"Daily output is missing columns: {sorted(missing_daily)}"
        )
    required_summary = {
        "zone_id",
        "planting_date",
        "emergence_date",
        "anthesis_date",
        "maturity_date",
        "LAIMAX",
        "TWSO",
        "TAGP",
        "NuptakeTotal",
        "total_biomass_kg_ha",
    }
    missing_summary = required_summary - set(summary.columns)
    if missing_summary:
        raise RuntimeError(
            f"Summary output is missing columns: {sorted(missing_summary)}"
        )

    if set(daily["zone_id"]) != expected_zone_ids:
        raise RuntimeError("Daily output does not contain every configured zone.")
    if set(summary["zone_id"]) != expected_zone_ids:
        raise RuntimeError("Summary output does not contain every configured zone.")
    if summary["zone_id"].duplicated().any():
        raise RuntimeError("Summary output contains duplicate zone rows.")
    if daily.duplicated(["zone_id", "date"]).any():
        raise RuntimeError("Daily output contains duplicate (zone_id, date) keys.")
    if summary["maturity_date"].isna().any():
        raise RuntimeError("At least one zone did not reach maturity.")

    nonnegative_variables = [
        "LAI",
        "TAGP",
        "WLV",
        "WST",
        "WRT",
        "WSO",
        "NAVAIL",
        "NNI",
        "NamountSO",
        "NamountLV",
        "NamountST",
        "NamountRT",
        "NuptakeTotal",
    ]
    for zone_id, zone_daily in daily.groupby("zone_id", sort=False):
        active = zone_daily.loc[zone_daily["crop_active"].astype(bool)]
        if active.empty:
            raise RuntimeError(f"Zone {zone_id} has no active crop days.")

        dvs = pd.to_numeric(active["DVS"], errors="coerce")
        if dvs.isna().any() or (~dvs.map(isfinite)).any():
            raise RuntimeError(f"Zone {zone_id} has invalid active-day DVS values.")
        if (dvs.diff().dropna() < -1e-8).any():
            raise RuntimeError(f"Zone {zone_id} has decreasing DVS.")

        for variable in nonnegative_variables:
            values = pd.to_numeric(active[variable], errors="coerce")
            if values.isna().any() or (~values.map(isfinite)).any():
                raise RuntimeError(
                    f"Zone {zone_id} has invalid active-day {variable} values."
                )
            if (values < -1e-8).any():
                raise RuntimeError(
                    f"Zone {zone_id} has negative active-day {variable} values."
                )

        moisture = pd.to_numeric(active["SM"], errors="coerce")
        lower = pd.to_numeric(active["soil_smw"], errors="coerce")
        upper = pd.to_numeric(active["soil_sm0"], errors="coerce")
        if moisture.isna().any() or ((moisture < lower - 1e-8) | (moisture > upper + 1e-8)).any():
            raise RuntimeError(f"Zone {zone_id} has soil moisture outside soil bounds.")

        zone_summary = summary.loc[summary["zone_id"] == zone_id].iloc[0]
        peak_lai = float(pd.to_numeric(active["LAI"]).max())
        terminal_wso = float(pd.to_numeric(active["WSO"]).dropna().iloc[-1])
        if not isclose(peak_lai, float(zone_summary["LAIMAX"]), rel_tol=1e-7, abs_tol=1e-7):
            raise RuntimeError(f"Zone {zone_id} LAIMAX does not match daily LAI.")
        if not isclose(terminal_wso, float(zone_summary["TWSO"]), rel_tol=1e-7, abs_tol=1e-7):
            raise RuntimeError(f"Zone {zone_id} TWSO does not match terminal WSO.")


def write_artifacts(
    config: Phase1Config, result: FarmResult, *, overwrite: bool = False
) -> tuple[Path, ...]:
    """Write validated CSVs and plots as one post-simulation batch."""

    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    output_directory = config.output.directory
    output_directory.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=".phase1-", dir=output_directory) as temporary:
        temporary_directory = Path(temporary)
        temporary_paths = tuple(
            temporary_directory / target.name for target in targets
        )

        result.daily.to_csv(
            temporary_paths[0],
            index=False,
            date_format="%Y-%m-%d",
            float_format="%.10g",
            lineterminator="\n",
        )
        result.summary.to_csv(
            temporary_paths[1],
            index=False,
            date_format="%Y-%m-%d",
            float_format="%.10g",
            lineterminator="\n",
        )
        _plot_trajectory(
            result.daily,
            variable="LAI",
            ylabel="Leaf area index (m2/m2)",
            title="Phase 1 LAI trajectories",
            path=temporary_paths[2],
        )
        _plot_trajectory(
            result.daily,
            variable="SM",
            ylabel="Volumetric soil moisture (cm3/cm3)",
            title="Phase 1 soil-moisture trajectories",
            path=temporary_paths[3],
        )
        _plot_nitrogen(result.daily, temporary_paths[4])
        _plot_yield_heatmap(result.summary, temporary_paths[5])

        for temporary_path, target in zip(temporary_paths, targets, strict=True):
            temporary_path.replace(target)

    return targets


def execute(config_path: str | Path, *, overwrite: bool = False) -> tuple[Path, ...]:
    """Validate inputs, run all zones, and write the Phase 1 artifacts."""

    config = load_phase1_config(config_path)
    farm = load_farm(config)
    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    validate_weather_file(config)
    result = run_farm(config, farm=farm)
    return write_artifacts(config, result, overwrite=overwrite)


def _parse_weather_metadata(lines: Iterable[str], path: Path) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for line in lines:
        for statement in line.split(";"):
            statement = statement.strip()
            if not statement or statement.startswith("#"):
                continue
            if "=" not in statement:
                raise ValueError(f"Invalid weather metadata in {path}: {statement}")
            key, raw_value = statement.split("=", 1)
            try:
                metadata[key.strip()] = ast.literal_eval(raw_value.strip())
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"Invalid weather metadata value for {key.strip()}."
                ) from exc
    return metadata


def _date_set(start: date, end: date) -> set[date]:
    return {
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
    }


def _artifact_paths(config: Phase1Config) -> tuple[Path, ...]:
    output = config.output
    paths = tuple(
        output.directory / filename
        for filename in (
            output.daily_truth_filename,
            output.zone_summary_filename,
            output.lai_plot_filename,
            output.soil_moisture_plot_filename,
            output.nitrogen_plot_filename,
            output.yield_heatmap_filename,
        )
    )
    if len(set(paths)) != len(paths):
        raise ValueError("Output artifact file names must be unique.")
    return paths


def _ensure_targets_available(
    targets: Sequence[Path], *, overwrite: bool
) -> None:
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to replace existing Phase 1 artifact(s): {rendered}. "
            "Pass --overwrite to replace them."
        )


def _plot_trajectory(
    daily: pd.DataFrame,
    *,
    variable: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    colors = {"low": "#b86b25", "reference": "#26734d", "high": "#3267a8"}
    figure, axis = plt.subplots(figsize=(11, 6))
    labelled: set[str] = set()
    for _, zone_daily in daily.groupby("zone_id", sort=False):
        profile = str(zone_daily["soil_profile"].iloc[0])
        label = profile if profile not in labelled else None
        labelled.add(profile)
        axis.plot(
            pd.to_datetime(zone_daily["date"]),
            zone_daily[variable],
            color=colors.get(profile, "#666666"),
            alpha=0.55,
            linewidth=1.2,
            label=label,
        )
    axis.set(title=title, xlabel="Date", ylabel=ylabel)
    axis.grid(alpha=0.2)
    axis.legend(title="Synthetic soil profile")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_nitrogen(daily: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for _, zone_daily in daily.groupby("zone_id", sort=False):
        dates = pd.to_datetime(zone_daily["date"])
        axes[0].plot(dates, zone_daily["NAVAIL"], alpha=0.45, linewidth=1.1)
        axes[1].plot(dates, zone_daily["NNI"], alpha=0.45, linewidth=1.1)
    axes[0].set(ylabel="Available N (kg N/ha)", title="Phase 1 nitrogen trajectories")
    axes[1].set(xlabel="Date", ylabel="Nitrogen nutrition index")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_yield_heatmap(summary: pd.DataFrame, path: Path) -> None:
    grid = summary.pivot(index="row", columns="column", values="TWSO").sort_index()
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(grid.to_numpy(), cmap="YlGn", origin="lower")
    axis.set(
        title="Final storage-organ biomass proxy (TWSO)",
        xlabel="Zone column",
        ylabel="Zone row",
        xticks=range(len(grid.columns)),
        yticks=range(len(grid.index)),
        xticklabels=grid.columns,
        yticklabels=grid.index,
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("kg dry matter/ha")
    for row_index in range(grid.shape[0]):
        for column_index in range(grid.shape[1]):
            value = float(grid.iloc[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 1 25-zone spatial ground-truth farm."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the Phase 1 YAML configuration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Phase 1 artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = execute(args.config, overwrite=args.overwrite)
    print("Phase 1 completed. Wrote:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
