"""Run controlled Phase 2 crop stresses against an unstressed baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from math import isclose
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.crop import TruthModel
from src.farm import Farm, load_farm
from src.simulation.run_phase1 import (
    FarmResult,
    run_farm,
    validate_farm_result,
    validate_weather_file,
)
from src.simulation.stress import (
    STRESS_TYPES,
    Phase2Config,
    StressEvent,
    events_by_zone,
    load_phase2_config,
)


@dataclass(frozen=True)
class Phase2Result:
    """Combined baseline, stressed, and causal-comparison outputs."""

    daily: pd.DataFrame
    summary: pd.DataFrame
    impacts: pd.DataFrame


def run_phase2(
    config: Phase2Config,
    *,
    farm: Farm | None = None,
    truth_model: TruthModel | None = None,
) -> Phase2Result:
    """Run both treatments completely without writing output files."""

    farm = farm or load_farm(config.phase1)
    truth_model = truth_model or TruthModel(config.phase1)
    grouped_events = events_by_zone(config.events)

    baseline = run_farm(config.phase1, farm=farm, truth_model=truth_model)
    stressed = _run_stressed_farm(
        config,
        farm=farm,
        truth_model=truth_model,
        grouped_events=grouped_events,
    )

    baseline_daily = _annotate_daily(
        baseline.daily, scenario="baseline", events_by_zone={}
    )
    stressed_daily = _annotate_daily(
        stressed.daily, scenario="stressed", events_by_zone=grouped_events
    )
    baseline_summary = _annotate_summary(
        baseline.summary, scenario="baseline", events_by_zone={}
    )
    stressed_summary = _annotate_summary(
        stressed.summary, scenario="stressed", events_by_zone=grouped_events
    )

    daily = pd.concat([baseline_daily, stressed_daily], ignore_index=True)
    summary = pd.concat([baseline_summary, stressed_summary], ignore_index=True)
    daily = _sort_scenarios(daily, ["zone_id", "date"])
    summary = _sort_scenarios(summary, ["zone_id"])
    impacts = _build_impacts(config.events, baseline, stressed)

    result = Phase2Result(daily=daily, summary=summary, impacts=impacts)
    validate_phase2_result(result, config=config, farm=farm)
    return result


def _run_stressed_farm(
    config: Phase2Config,
    *,
    farm: Farm,
    truth_model: TruthModel,
    grouped_events: dict[str, tuple[StressEvent, ...]],
) -> FarmResult:
    daily_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for zone in sorted(farm.zones, key=lambda item: (item.row, item.column)):
        soil = config.phase1.soil_profiles[zone.soil_profile]
        zone_result = truth_model.run_zone(
            zone, soil, stress_events=grouped_events.get(zone.zone_id, ())
        )
        daily_frames.append(zone_result.daily)
        summaries.append(zone_result.summary)

    daily = pd.concat(daily_frames, ignore_index=True).sort_values(
        ["row", "column", "date"]
    ).reset_index(drop=True)
    summary = pd.DataFrame.from_records(summaries).sort_values(
        ["row", "column"]
    ).reset_index(drop=True)
    summary["total_biomass_kg_ha"] = summary[
        ["TWLV", "TWST", "TWRT", "TWSO"]
    ].sum(axis=1, min_count=4)
    result = FarmResult(daily=daily, summary=summary)
    validate_farm_result(result, farm)
    return result


def _annotate_daily(
    daily: pd.DataFrame,
    *,
    scenario: str,
    events_by_zone: dict[str, tuple[StressEvent, ...]],
) -> pd.DataFrame:
    annotated = daily.copy()
    annotated.insert(0, "scenario", scenario)
    assigned: list[str] = []
    active: list[str] = []
    for row in annotated.itertuples(index=False):
        events = events_by_zone.get(row.zone_id, ())
        assigned.append("|".join(event.event_id for event in events))
        active.append(
            "|".join(
                event.event_id
                for event in events
                if event.is_active(row.date, row.planting_date)
            )
        )
    annotated.insert(1, "assigned_event_ids", assigned)
    annotated.insert(2, "active_event_ids", active)
    annotated.insert(3, "stress_active", [bool(value) for value in active])
    return annotated


def _annotate_summary(
    summary: pd.DataFrame,
    *,
    scenario: str,
    events_by_zone: dict[str, tuple[StressEvent, ...]],
) -> pd.DataFrame:
    annotated = summary.copy()
    annotated.insert(0, "scenario", scenario)
    annotated.insert(
        1,
        "assigned_event_ids",
        [
            "|".join(
                event.event_id for event in events_by_zone.get(zone_id, ())
            )
            for zone_id in annotated["zone_id"]
        ],
    )
    return annotated


def _build_impacts(
    events: Sequence[StressEvent], baseline: FarmResult, stressed: FarmResult
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    baseline_summary = baseline.summary.set_index("zone_id")
    stressed_summary = stressed.summary.set_index("zone_id")
    for event in events:
        before = baseline_summary.loc[event.zone_id]
        after = stressed_summary.loc[event.zone_id]
        planting_date = before["planting_date"]
        baseline_twso = float(before["TWSO"])
        stressed_twso = float(after["TWSO"])
        yield_loss = baseline_twso - stressed_twso
        records.append(
            {
                "event_id": event.event_id,
                "stress_type": event.stress_type,
                "zone_id": event.zone_id,
                "row": int(before["row"]),
                "column": int(before["column"]),
                "start_day": event.start_day,
                "duration_days": event.duration_days,
                "onset_date": event.start_date(planting_date),
                "end_date": event.end_date(planting_date),
                "severity": event.severity,
                "first_lai_divergence_date": _first_lai_divergence(
                    event.zone_id, baseline.daily, stressed.daily
                ),
                "baseline_LAIMAX": float(before["LAIMAX"]),
                "stressed_LAIMAX": float(after["LAIMAX"]),
                "lai_max_delta": float(after["LAIMAX"] - before["LAIMAX"]),
                "baseline_TWSO": baseline_twso,
                "stressed_TWSO": stressed_twso,
                "yield_loss_kg_ha": yield_loss,
                "yield_loss_pct": 100.0 * yield_loss / baseline_twso,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["event_id", "row", "column"]
    ).reset_index(drop=True)


def _first_lai_divergence(
    zone_id: str, baseline: pd.DataFrame, stressed: pd.DataFrame
) -> date | None:
    left = baseline.loc[baseline["zone_id"] == zone_id, ["date", "LAI"]]
    right = stressed.loc[stressed["zone_id"] == zone_id, ["date", "LAI"]]
    comparison = left.merge(right, on="date", suffixes=("_baseline", "_stressed"))
    difference = (
        pd.to_numeric(comparison["LAI_stressed"], errors="coerce")
        - pd.to_numeric(comparison["LAI_baseline"], errors="coerce")
    ).abs()
    changed = comparison.loc[difference > 1e-8, "date"]
    return None if changed.empty else changed.iloc[0]


def validate_phase2_result(
    result: Phase2Result, *, config: Phase2Config, farm: Farm
) -> None:
    """Validate paired scenarios and the measurable Phase 2 event effects."""

    expected_zones = {zone.zone_id for zone in farm.zones}
    if set(result.daily["scenario"]) != {"baseline", "stressed"}:
        raise RuntimeError("Daily output must contain baseline and stressed scenarios.")
    if set(result.summary["scenario"]) != {"baseline", "stressed"}:
        raise RuntimeError("Summary output must contain baseline and stressed scenarios.")
    if result.daily.duplicated(["scenario", "zone_id", "date"]).any():
        raise RuntimeError("Daily output contains duplicate scenario/zone/date keys.")
    if result.summary.duplicated(["scenario", "zone_id"]).any():
        raise RuntimeError("Summary output contains duplicate scenario/zone rows.")

    for scenario in ("baseline", "stressed"):
        zone_ids = set(
            result.summary.loc[result.summary["scenario"] == scenario, "zone_id"]
        )
        if zone_ids != expected_zones:
            raise RuntimeError(f"{scenario} summary does not contain every zone.")

    baseline = result.summary.loc[result.summary["scenario"] == "baseline"].set_index(
        "zone_id"
    )
    stressed = result.summary.loc[result.summary["scenario"] == "stressed"].set_index(
        "zone_id"
    )
    affected_zones = {event.zone_id for event in config.events}
    for zone_id in expected_zones - affected_zones:
        for variable in ("LAIMAX", "TWSO", "NuptakeTotal"):
            if not isclose(
                float(baseline.loc[zone_id, variable]),
                float(stressed.loc[zone_id, variable]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise RuntimeError(
                    f"Unaffected zone {zone_id} changed {variable} between scenarios."
                )

    expected_pairs = {(event.event_id, event.zone_id) for event in config.events}
    actual_pairs = set(zip(result.impacts["event_id"], result.impacts["zone_id"]))
    if actual_pairs != expected_pairs:
        raise RuntimeError("Stress impacts do not match the configured event footprint.")
    if result.impacts["first_lai_divergence_date"].isna().any():
        raise RuntimeError("At least one stress event has no measurable LAI effect.")
    if (result.impacts["yield_loss_kg_ha"] <= 1e-6).any():
        raise RuntimeError("At least one stress event has no measurable yield loss.")

    stressed_daily = result.daily.loc[result.daily["scenario"] == "stressed"]
    represented = {
        event_id
        for values in stressed_daily["active_event_ids"]
        for event_id in str(values).split("|")
        if event_id
    }
    expected_event_ids = {event.event_id for event in config.events}
    if represented != expected_event_ids:
        raise RuntimeError("Daily output does not mark every event's active period.")


def write_artifacts(
    config: Phase2Config, result: Phase2Result, *, overwrite: bool = False
) -> tuple[Path, ...]:
    """Write Phase 2 tables and plots only after all validation succeeds."""

    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    config.output.directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase2-", dir=config.output.directory) as temporary:
        temporary_paths = tuple(Path(temporary) / target.name for target in targets)
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
        result.impacts.to_csv(
            temporary_paths[2],
            index=False,
            date_format="%Y-%m-%d",
            float_format="%.10g",
            lineterminator="\n",
        )
        _plot_stress_trajectories(result, temporary_paths[3])
        _plot_yield_impact(result, farm=load_farm(config.phase1), path=temporary_paths[4])
        for temporary_path, target in zip(temporary_paths, targets, strict=True):
            temporary_path.replace(target)
    return targets


def execute(config_path: str | Path, *, overwrite: bool = False) -> tuple[Path, ...]:
    config = load_phase2_config(config_path)
    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    validate_weather_file(config.phase1)
    farm = load_farm(config.phase1)
    result = run_phase2(config, farm=farm)
    return write_artifacts(config, result, overwrite=overwrite)


def _sort_scenarios(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    sorted_frame = frame.assign(
        _scenario_order=frame["scenario"].map({"baseline": 0, "stressed": 1})
    ).sort_values(["_scenario_order", *columns])
    return sorted_frame.drop(columns="_scenario_order").reset_index(drop=True)


def _artifact_paths(config: Phase2Config) -> tuple[Path, ...]:
    output = config.output
    paths = tuple(
        output.directory / filename
        for filename in (
            output.daily_truth_filename,
            output.zone_summary_filename,
            output.stress_impacts_filename,
            output.trajectory_plot_filename,
            output.yield_impact_heatmap_filename,
        )
    )
    if len(set(paths)) != len(paths):
        raise ValueError("Phase 2 output artifact file names must be unique.")
    return paths


def _ensure_targets_available(targets: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to replace existing Phase 2 artifact(s): {rendered}. "
            "Pass --overwrite to replace them."
        )


def _plot_stress_trajectories(result: Phase2Result, path: Path) -> None:
    figure, axes = plt.subplots(len(STRESS_TYPES), 1, figsize=(11, 10), sharex=True)
    for axis, stress_type in zip(axes, STRESS_TYPES, strict=True):
        impacts = result.impacts.loc[result.impacts["stress_type"] == stress_type]
        for color_index, zone_id in enumerate(impacts["zone_id"]):
            for scenario, style in (("baseline", "--"), ("stressed", "-")):
                rows = result.daily.loc[
                    (result.daily["scenario"] == scenario)
                    & (result.daily["zone_id"] == zone_id)
                ]
                axis.plot(
                    pd.to_datetime(rows["date"]),
                    rows["LAI"],
                    color=f"C{color_index}",
                    linestyle=style,
                    linewidth=1.5,
                    label=f"{zone_id} {scenario}",
                )
        axis.set(title=stress_type.replace("_", " ").title(), ylabel="LAI")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Date")
    figure.suptitle("Phase 2 baseline versus stressed LAI")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_yield_impact(result: Phase2Result, *, farm: Farm, path: Path) -> None:
    values = pd.DataFrame(
        0.0,
        index=range(1, farm.rows + 1),
        columns=range(1, farm.columns + 1),
    )
    for impact in result.impacts.itertuples(index=False):
        values.loc[impact.row, impact.column] = impact.yield_loss_pct

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values.to_numpy(), cmap="OrRd", origin="lower", vmin=0)
    axis.set(
        title="Phase 2 yield-proxy loss by stressed zone",
        xlabel="Zone column",
        ylabel="Zone row",
        xticks=range(farm.columns),
        yticks=range(farm.rows),
        xticklabels=values.columns,
        yticklabels=values.index,
    )
    figure.colorbar(image, ax=axis).set_label("TWSO loss (%)")
    for row_index in range(farm.rows):
        for column_index in range(farm.columns):
            axis.text(
                column_index,
                row_index,
                f"{values.iloc[row_index, column_index]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired baseline and controlled-stress Phase 2 scenarios."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Phase 2 artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = execute(args.config, overwrite=args.overwrite)
    print("Phase 2 completed. Wrote:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
