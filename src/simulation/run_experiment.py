"""CLI and atomic artifact writer for modular monitoring experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping, Sequence

import pandas as pd

from src.crop import TruthModel
from src.farm import load_farm
from src.simulation.analyze_experiment import PLOT_FILENAMES, write_experiment_plots
from src.simulation.experiment import DistributionSummary, ExperimentResult, PairwiseResult, ScenarioMethodResult, SensitivityResult, run_experiment
from src.simulation.profiles import ExperimentConfig, load_experiment_config
from src.simulation.run_phase1 import run_farm, validate_weather_file


def execute(config_path: str | Path, *, overwrite: bool = False) -> tuple[Path, ...]:
    config = load_experiment_config(config_path)
    targets = _artifact_paths(config)
    _ensure_available(targets, overwrite)
    validate_weather_file(config.phase1)
    farm = load_farm(config.phase1)
    truth = run_farm(config.phase1, farm=farm, truth_model=TruthModel(config.phase1))
    baseline = {str(row["zone_id"]): float(row["TWSO"]) for row in truth.summary.to_dict(orient="records")}
    result = run_experiment(config, truth.daily, farm=farm, baseline_twso_kg_ha=baseline)
    return write_artifacts(config, result, overwrite=overwrite)


def write_artifacts(config: ExperimentConfig, result: ExperimentResult, *, overwrite: bool = False) -> tuple[Path, ...]:
    targets = _artifact_paths(config)
    _ensure_available(targets, overwrite)
    config.output.directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".experiment-", dir=config.output.directory) as temporary:
        root = Path(temporary)
        csv_paths = [root / target.name for target in targets[:4]]
        _write_csv(csv_paths[0], result.method_results, tuple(ScenarioMethodResult.__dataclass_fields__))
        _write_csv(csv_paths[1], result.pairwise_results, tuple(PairwiseResult.__dataclass_fields__))
        _write_csv(csv_paths[2], result.distributions, tuple(DistributionSummary.__dataclass_fields__))
        _write_csv(csv_paths[3], result.sensitivities, tuple(SensitivityResult.__dataclass_fields__))
        manifest_path = root / targets[4].name
        manifest_path.write_text(json.dumps(_manifest(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        plot_paths = write_experiment_plots(pd.DataFrame(asdict(item) for item in result.method_results), pd.DataFrame(asdict(item) for item in result.pairwise_results), pd.DataFrame(asdict(item) for item in result.sensitivities), root)
        temporary_paths = (*csv_paths, manifest_path, *plot_paths)
        for temporary_path, target in zip(temporary_paths, targets, strict=True):
            temporary_path.replace(target)
    return targets


def _manifest(config: ExperimentConfig) -> Mapping[str, object]:
    try:
        pcse_version = importlib.metadata.version("PCSE")
    except importlib.metadata.PackageNotFoundError:
        pcse_version = "unavailable"
    return {
        "claim_boundary": "Synthetic sensitivity evidence under configured assumptions; not field validation, commercial ROI, or causal proof.",
        "config": str(config.source_path),
        "scenario_count": config.scenario_count,
        "seed": config.seed,
        "reference_method_id": config.reference_method_id,
        "methods": [{"id": item.method_id, "kind": item.profile.kind, "family": item.profile.family} for item in config.methods],
        "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in config.input_paths],
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "pcse": pcse_version},
    }


def _artifact_paths(config: ExperimentConfig) -> tuple[Path, ...]:
    output = config.output
    names = (output.method_results_filename, output.pairwise_results_filename, output.distribution_summary_filename, output.sensitivity_filename, output.manifest_filename, *PLOT_FILENAMES)
    paths = tuple(output.directory / name for name in names)
    if len(paths) != len(set(paths)):
        raise ValueError("experiment output filenames must be unique")
    return paths


def _ensure_available(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to replace existing experiment artifacts: {existing}")


def _write_csv(path: Path, rows: Iterable[object], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a modular crop-monitoring experiment.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    for path in execute(args.config, overwrite=args.overwrite):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
