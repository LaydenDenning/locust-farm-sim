"""Command-line runner for Phase 10 provisional economics."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping, Sequence

from src.farm import load_farm
from src.simulation.actions import simulate_actions
from src.simulation.economics import (
    Phase10Config,
    Phase10Result,
    StrategyEconomics,
    ZoneEconomics,
    calculate_economics,
    load_phase10_config,
)
from src.simulation.run_phase1 import run_farm, validate_weather_file
from src.simulation.run_phase8 import run_phase8


def run_phase10(config: Phase10Config) -> Phase10Result:
    """Run the decision chain and value its Phase 9 outcomes."""

    phase8 = run_phase8(config.phase9.phase8)
    phase9 = simulate_actions(config.phase9, phase8)
    phase1 = config.phase9.phase8.phase7.phase5.phase4.phase1
    farm = load_farm(phase1)
    baseline = run_farm(phase1, farm=farm).summary
    twso = {
        str(row["zone_id"]): float(row["TWSO"])
        for row in baseline.to_dict(orient="records")
    }
    return calculate_economics(config, phase8, phase9, farm=farm, baseline_twso_kg_ha=twso)


def write_artifacts(
    config: Phase10Config, result: Phase10Result, *, overwrite: bool = False
) -> tuple[Path, ...]:
    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    config.output.directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase10-", dir=config.output.directory) as temporary:
        temporary_paths = tuple(Path(temporary) / target.name for target in targets)
        _write_csv(
            temporary_paths[0],
            _records(result.zone_economics),
            tuple(ZoneEconomics.__dataclass_fields__),
        )
        _write_csv(
            temporary_paths[1],
            _records(result.strategy_summary),
            tuple(StrategyEconomics.__dataclass_fields__),
        )
        for temporary_path, target in zip(temporary_paths, targets, strict=True):
            temporary_path.replace(target)
    return targets


def execute(config_path: str | Path, *, overwrite: bool = False) -> tuple[Path, ...]:
    config = load_phase10_config(config_path)
    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    validate_weather_file(config.phase9.phase8.phase7.phase5.phase4.phase1)
    return write_artifacts(config, run_phase10(config), overwrite=overwrite)


def _records(items: Iterable[object]) -> Iterable[Mapping[str, object]]:
    for item in items:
        record = asdict(item)
        for key, value in tuple(record.items()):
            if hasattr(value, "isoformat"):
                record[key] = value.isoformat()
        yield record


def _write_csv(
    path: Path,
    records: Iterable[Mapping[str, object]],
    fieldnames: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def _artifact_paths(config: Phase10Config) -> tuple[Path, ...]:
    paths = (
        config.output.directory / config.output.zone_economics_filename,
        config.output.directory / config.output.strategy_summary_filename,
    )
    if len(set(paths)) != len(paths):
        raise ValueError("Phase 10 output artifact file names must be unique")
    return paths


def _ensure_targets_available(targets: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to replace existing Phase 10 artifact(s): {rendered}. "
            "Pass --overwrite to replace them."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate provisional Phase 10 TWSO-proxy economics."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing Phase 10 artifacts."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = execute(args.config, overwrite=args.overwrite)
    print("Phase 10 completed. Wrote:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
