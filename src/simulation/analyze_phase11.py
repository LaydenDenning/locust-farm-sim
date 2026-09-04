"""Create readable plots from Phase 11 scenario outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCENARIO_FILENAME = "scenario_results.csv"
SENSITIVITY_FILENAME = "sensitivity.csv"
PLOT_FILENAMES = (
    "strategy_net_benefits.png",
    "strategy_comparison.png",
    "drone_advantage.png",
    "sensitivity_correlations.png",
)

SCENARIO_COLUMNS = (
    "mechanism",
    "drone_avoided_twso_kg",
    "scout_avoided_twso_kg",
    "drone_net_benefit",
    "scout_net_benefit",
    "drone_advantage_vs_scout",
)
SENSITIVITY_COLUMNS = (
    "parameter",
    "correlation_with_drone_advantage",
)


def analyze_phase11(
    input_directory: str | Path,
    output_directory: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Read Phase 11 CSVs and write four analysis plots."""

    input_path = Path(input_directory)
    output_path = Path(output_directory) if output_directory else input_path
    targets = tuple(output_path / name for name in PLOT_FILENAMES)
    _ensure_targets_available(targets, overwrite=overwrite)

    scenarios = _read_csv(
        input_path / SCENARIO_FILENAME,
        required_columns=SCENARIO_COLUMNS,
        numeric_columns=SCENARIO_COLUMNS[1:],
    )
    sensitivities = _read_csv(
        input_path / SENSITIVITY_FILENAME,
        required_columns=SENSITIVITY_COLUMNS,
        numeric_columns=SENSITIVITY_COLUMNS[1:],
    )

    output_path.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase11-analysis-", dir=output_path) as temporary:
        temporary_paths = tuple(Path(temporary) / path.name for path in targets)
        _plot_strategy_distributions(scenarios, temporary_paths[0])
        _plot_strategy_comparison(scenarios, temporary_paths[1])
        _plot_drone_advantage(scenarios, temporary_paths[2])
        _plot_sensitivities(sensitivities, temporary_paths[3])
        for temporary_path, target in zip(temporary_paths, targets, strict=True):
            temporary_path.replace(target)
    return targets


def _read_csv(
    path: Path,
    *,
    required_columns: Sequence[str],
    numeric_columns: Sequence[str],
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Unable to read {path}: {exc}") from exc

    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} contains no data rows")

    frame = frame.loc[:, required_columns].copy()
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"{path} contains invalid values in {column}")
    return frame


def _ensure_targets_available(targets: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to replace existing analysis plot(s): {rendered}. "
            "Pass --overwrite to replace them."
        )


def _plot_strategy_distributions(scenarios: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    _plot_strategy_intervals(
        axes[0],
        scenarios,
        columns=("drone_avoided_twso_kg", "scout_avoided_twso_kg"),
        title="Crop protection",
        xlabel="Avoided TWSO-proxy loss (kg)\nHigher is better",
    )
    _plot_strategy_intervals(
        axes[1],
        scenarios,
        columns=("drone_net_benefit", "scout_net_benefit"),
        title="Financial return after configured costs",
        xlabel="Net benefit versus no intervention\n(configured currency units)",
    )
    figure.suptitle("Phase 11 outcomes: crop protection versus financial return")
    figure.text(
        0.5,
        0.015,
        "Points are medians; lines span the 5th–95th percentiles. "
        "Net benefit includes operating and treatment costs.",
        ha="center",
        color="dimgray",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_strategy_intervals(
    axis: plt.Axes,
    scenarios: pd.DataFrame,
    *,
    columns: tuple[str, str],
    title: str,
    xlabel: str,
) -> None:
    labels = ("Drone", "Scout")
    colors = ("C0", "C1")
    for position, (column, label, color) in enumerate(
        zip(columns, labels, colors, strict=True)
    ):
        values = scenarios[column]
        low = float(values.quantile(0.05))
        median = float(values.median())
        high = float(values.quantile(0.95))
        positive_rate = float((values > 0).mean())
        axis.errorbar(
            median,
            position,
            xerr=((median - low,), (high - median,)),
            fmt="o",
            color=color,
            capsize=5,
            markersize=8,
            linewidth=2,
        )
        axis.annotate(
            f"median {median:,.0f}\n{positive_rate:.0%} above zero",
            (median, position),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.axvline(0, color="black", linewidth=1, linestyle="--")
    axis.set(
        title=title,
        xlabel=xlabel,
        yticks=range(len(labels)),
        yticklabels=labels,
        ylim=(-0.5, len(labels) - 0.5),
    )
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.2)


def _plot_strategy_comparison(scenarios: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 8))
    values = pd.concat(
        [scenarios["scout_net_benefit"], scenarios["drone_net_benefit"]]
    )
    lower = min(float(values.min()), 0.0)
    upper = max(float(values.max()), 0.0)
    if lower == upper:
        lower -= 1
        upper += 1
    padding = (upper - lower) * 0.05
    lower -= padding
    upper += padding
    zero_position = (0 - lower) / (upper - lower)

    axis.axvspan(lower, 0, ymin=zero_position, color="C0", alpha=0.06)
    axis.axvspan(0, upper, ymin=zero_position, color="C2", alpha=0.08)
    axis.axvspan(lower, 0, ymax=zero_position, color="gray", alpha=0.08)
    axis.axvspan(0, upper, ymax=zero_position, color="C1", alpha=0.06)
    for mechanism, rows in scenarios.groupby("mechanism", sort=True):
        axis.scatter(
            rows["scout_net_benefit"],
            rows["drone_net_benefit"],
            alpha=0.75,
            label=str(mechanism).replace("_", " ").title(),
            zorder=3,
        )

    axis.axhline(0, color="black", linewidth=1)
    axis.axvline(0, color="black", linewidth=1)
    axis.plot(
        [lower, upper],
        [lower, upper],
        color="black",
        linewidth=1.3,
        linestyle="--",
    )
    axis.set(
        title=(
            "Financial comparison by scenario\n"
            "Above the dashed line, the drone performs better than scouting"
        ),
        xlabel="Scout net benefit versus no intervention",
        ylabel="Drone net benefit versus no intervention",
        xlim=(lower, upper),
        ylim=(lower, upper),
    )
    axis.text(
        0.02,
        0.98,
        "Only drone beats\nno intervention",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
    )
    axis.text(
        0.98,
        0.98,
        "Both beat\nno intervention",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    axis.text(
        0.02,
        0.02,
        "Neither beats\nno intervention",
        transform=axis.transAxes,
        va="bottom",
        fontsize=8,
    )
    axis.text(
        0.98,
        0.02,
        "Only scouting beats\nno intervention",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    scenario_count = len(scenarios)
    drone_positive = int((scenarios["drone_net_benefit"] > 0).sum())
    drone_better = int((scenarios["drone_advantage_vs_scout"] > 0).sum())
    axis.text(
        0.98,
        0.52,
        f"Drone beats no intervention: {drone_positive}/{scenario_count}\n"
        f"Drone beats scouting: {drone_better}/{scenario_count}",
        transform=axis.transAxes,
        ha="right",
        va="center",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )
    axis.grid(alpha=0.2)
    axis.legend(
        title="Issue mechanism",
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    figure.text(
        0.5,
        0.015,
        "Zero on either axis is the no-intervention financial baseline. "
        "Synthetic scenarios under configured assumptions.",
        ha="center",
        color="dimgray",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 0.84, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_drone_advantage(scenarios: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    bins = max(5, min(20, round(len(scenarios) ** 0.5)))
    values = scenarios["drone_advantage_vs_scout"]
    axis.hist(
        [values.loc[values < 0], values.loc[values >= 0]],
        bins=bins,
        color=["C3", "C2"],
        label=["Scouting financially better", "Drone financially better"],
        stacked=True,
        alpha=0.8,
    )
    axis.axvline(0, color="black", linewidth=1, linestyle="--")
    median = float(values.median())
    positive = int((values > 0).sum())
    axis.set(
        title="Financial advantage: drone versus scouting\n"
        "Drone advantage = drone net benefit − scout net benefit",
        xlabel="Difference in net benefit (configured currency units)",
        ylabel="Scenario count",
    )
    axis.text(
        0.98,
        0.95,
        f"Drone financially better: {positive}/{len(values)} scenarios\n"
        f"Median advantage: {median:,.0f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.text(
        0.5,
        0.015,
        "Negative values favor scouting; positive values favor the drone. "
        "Synthetic scenarios under configured assumptions.",
        ha="center",
        color="dimgray",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_sensitivities(sensitivities: pd.DataFrame, path: Path) -> None:
    rows = sensitivities.assign(
        absolute_correlation=sensitivities[
            "correlation_with_drone_advantage"
        ].abs()
    ).sort_values("absolute_correlation")
    values = rows["correlation_with_drone_advantage"]
    labels = (
        rows["parameter"]
        .astype(str)
        .str.replace("_", " ")
        .str.replace("twso proxy", "crop-value proxy")
    )
    colors = ["C2" if value >= 0 else "C3" for value in values]

    figure, axis = plt.subplots(figsize=(10, max(5, len(rows) * 0.45)))
    bars = axis.barh(labels, values, color=colors, alpha=0.8)
    axis.bar_label(bars, fmt="%+.2f", padding=3, fontsize=8)
    axis.axvline(0, color="black", linewidth=1)
    axis.set(
        title="Which inputs are associated with drone financial advantage?",
        xlabel=(
            "Correlation with drone − scout net benefit\n"
            "← increasing value is associated with less drone advantage   |   "
            "more drone advantage →"
        ),
        xlim=(-1, 1),
    )
    axis.grid(axis="x", alpha=0.2)
    figure.text(
        0.5,
        0.01,
        "Bars show association, not causation. Synthetic scenarios under "
        "configured assumptions.",
        ha="center",
        color="dimgray",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create plots from Phase 11 scenario CSVs."
    )
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=Path("outputs/phase11"),
        help="Directory containing scenario_results.csv and sensitivity.csv.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Plot destination; defaults to the input directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing analysis plots.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = analyze_phase11(
        args.input_directory,
        args.output_directory,
        overwrite=args.overwrite,
    )
    print("Phase 11 analysis completed. Wrote:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
