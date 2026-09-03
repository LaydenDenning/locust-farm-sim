"""Command-line runner for Phase 11 repeated paired scenarios."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping, Sequence

from src.farm import load_farm
from src.simulation.run_phase1 import run_farm, validate_weather_file
from src.simulation.scenarios import (
    DistributionSummary,
    Phase11Config,
    Phase11Result,
    ScenarioResult,
    SensitivityResult,
    load_phase11_config,
    run_scenarios,
)


def run_phase11(config: Phase11Config) -> Phase11Result:
    """Run WOFOST once, then reuse its baseline for every scenario."""

    phase1 = config.phase10.phase9.phase8.phase7.phase5.phase4.phase1
    farm = load_farm(phase1)
    truth = run_farm(phase1, farm=farm)
    baseline = {
        str(row["zone_id"]): float(row["TWSO"])
        for row in truth.summary.to_dict(orient="records")
    }
    return run_scenarios(
        config,
        truth.daily,
        farm=farm,
        baseline_twso_kg_ha=baseline,
    )


def write_artifacts(
    config: Phase11Config, result: Phase11Result, *, overwrite: bool = False
) -> tuple[Path, ...]:
    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    config.output.directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase11-", dir=config.output.directory) as temporary:
        temporary_paths = tuple(Path(temporary) / target.name for target in targets)
        _write_csv(temporary_paths[0], _records(result.scenarios), tuple(ScenarioResult.__dataclass_fields__))
        _write_csv(temporary_paths[1], _records(result.distributions), tuple(DistributionSummary.__dataclass_fields__))
        _write_csv(temporary_paths[2], _records(result.sensitivities), tuple(SensitivityResult.__dataclass_fields__))
        for temporary_path, target in zip(temporary_paths, targets, strict=True):
            temporary_path.replace(target)
    return targets


def execute(config_path: str | Path, *, overwrite: bool = False) -> tuple[Path, ...]:
    config = load_phase11_config(config_path)
    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    validate_weather_file(config.phase10.phase9.phase8.phase7.phase5.phase4.phase1)
    return write_artifacts(config, run_phase11(config), overwrite=overwrite)


def _records(items: Iterable[object]) -> Iterable[Mapping[str, object]]:
    return (asdict(item) for item in items)


def _write_csv(path: Path, records: Iterable[Mapping[str, object]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _artifact_paths(config: Phase11Config) -> tuple[Path, ...]:
    paths = (
        config.output.directory / config.output.scenario_results_filename,
        config.output.directory / config.output.distribution_summary_filename,
        config.output.directory / config.output.sensitivity_filename,
    )
    if len(set(paths)) != len(paths):
        raise ValueError("Phase 11 output artifact file names must be unique")
    return paths


def _ensure_targets_available(targets: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to replace existing Phase 11 artifact(s): {rendered}. Pass --overwrite to replace them."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 11 repeated paired scenarios.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing Phase 11 artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = execute(args.config, overwrite=args.overwrite)
    print("Phase 11 completed. Wrote:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
