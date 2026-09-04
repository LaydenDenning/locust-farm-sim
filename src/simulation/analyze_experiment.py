"""Method-agnostic plots for modular experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PLOT_FILENAMES = (
    "detection_performance.png",
    "agronomic_outcomes.png",
    "economic_outcomes.png",
    "pairwise_advantages.png",
    "sensitivity_correlations.png",
)


def write_experiment_plots(
    methods: pd.DataFrame,
    pairwise: pd.DataFrame,
    sensitivities: pd.DataFrame,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = tuple(output / name for name in PLOT_FILENAMES)
    _plot_detection(methods, paths[0])
    _plot_box(methods, "avoided_twso_kg", "Avoided TWSO-proxy loss", "kg", paths[1])
    _plot_box(methods, "net_benefit_vs_no_intervention", "Net benefit versus no intervention", "configured currency units", paths[2], zero=True)
    _plot_pairwise(pairwise, paths[3])
    _plot_sensitivity(sensitivities, paths[4])
    return paths


def analyze_experiment(
    input_directory: str | Path,
    output_directory: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    source = Path(input_directory)
    output = Path(output_directory) if output_directory else source
    targets = tuple(output / name for name in PLOT_FILENAMES)
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to replace existing plots: {existing}")
    methods = pd.read_csv(source / "scenario_method_results.csv")
    pairwise = pd.read_csv(source / "pairwise_results.csv")
    sensitivities = pd.read_csv(source / "sensitivity.csv")
    with TemporaryDirectory(prefix=".experiment-analysis-", dir=output) as temporary:
        temporary_paths = write_experiment_plots(methods, pairwise, sensitivities, temporary)
        output.mkdir(parents=True, exist_ok=True)
        for temporary_path, target in zip(temporary_paths, targets, strict=True):
            temporary_path.replace(target)
    return targets


def _plot_detection(frame: pd.DataFrame, path: Path) -> None:
    grouped = frame.groupby("method_id", sort=True)[["true_positives", "false_negatives", "false_positives", "unavailable_observations"]].sum()
    denominators = (grouped["true_positives"] + grouped["false_negatives"]).replace(0, 1)
    display = pd.DataFrame({"Detection recall": grouped["true_positives"] / denominators, "False-positive count / scenario": grouped["false_positives"] / frame["scenario_id"].nunique(), "Unavailable / scenario": grouped["unavailable_observations"] / frame["scenario_id"].nunique()})
    axis = display.plot.bar(figsize=(11, 6))
    axis.set(title="Detection and availability by monitoring method", ylabel="Rate or count", xlabel="Method")
    axis.grid(axis="y", alpha=0.2)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=150)
    plt.close(axis.figure)


def _plot_box(frame: pd.DataFrame, column: str, title: str, ylabel: str, path: Path, zero: bool = False) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    ids = sorted(frame["method_id"].unique())
    axis.boxplot([frame.loc[frame["method_id"] == method_id, column] for method_id in ids], tick_labels=ids, showfliers=False)
    if zero:
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set(title=title, ylabel=ylabel, xlabel="Method")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_pairwise(frame: pd.DataFrame, path: Path) -> None:
    comparisons = frame.loc[frame["reference_method_id"] != "no_intervention"].copy()
    comparisons["comparison"] = comparisons["candidate_method_id"] + " vs " + comparisons["reference_method_id"]
    figure, axis = plt.subplots(figsize=(10, 6))
    ids = sorted(comparisons["comparison"].unique())
    axis.boxplot([comparisons.loc[comparisons["comparison"] == item, "net_benefit_delta"] for item in ids], tick_labels=ids, showfliers=False)
    axis.axhline(0, color="black", linestyle="--", linewidth=1)
    axis.set(title="Paired financial advantage", ylabel="Net-benefit difference", xlabel="Comparison")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_sensitivity(frame: pd.DataFrame, path: Path) -> None:
    rows = frame.loc[frame["scope"] == "comparison"].copy()
    if rows.empty:
        rows = frame.copy()
    rows["absolute"] = rows["correlation"].abs()
    rows = rows.sort_values("absolute").groupby("result_id", sort=True).tail(6)
    labels = rows["result_id"] + ": " + rows["parameter"]
    colors = ["C2" if value >= 0 else "C3" for value in rows["correlation"]]
    figure, axis = plt.subplots(figsize=(11, max(5, len(rows) * 0.32)))
    axis.barh(labels, rows["correlation"], color=colors)
    axis.axvline(0, color="black", linewidth=1)
    axis.set(title="Strongest configured sensitivity associations", xlabel="Pearson correlation (association, not causation)", xlim=(-1, 1))
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot modular monitoring experiment outputs.")
    parser.add_argument("--input-directory", type=Path, default=Path("outputs/experiment"))
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    for path in analyze_experiment(args.input_directory, args.output_directory, overwrite=args.overwrite):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
