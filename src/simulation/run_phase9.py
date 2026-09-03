"""Command-line runner for Phase 9 treatment actions and outcomes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping, Sequence

from src.simulation.actions import (
    ActionEvaluation,
    ActionEvent,
    Phase9Config,
    Phase9Result,
    ScenarioOutcome,
    load_phase9_config,
    simulate_actions,
)
from src.simulation.run_phase1 import validate_weather_file
from src.simulation.run_phase8 import run_phase8


def run_phase9(config: Phase9Config) -> Phase9Result:
    """Run through Phase 8, then apply the Phase 9 action policy."""

    return simulate_actions(config, run_phase8(config.phase8))


def write_artifacts(
    config: Phase9Config, result: Phase9Result, *, overwrite: bool = False
) -> tuple[Path, ...]:
    """Write Phase 9 operational and evaluation artifacts atomically."""

    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    config.output.directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase9-", dir=config.output.directory) as temporary:
        temporary_paths = tuple(Path(temporary) / target.name for target in targets)
        _write_csv(
            temporary_paths[0],
            _records(result.action_events),
            tuple(ActionEvent.__dataclass_fields__),
        )
        _write_csv(
            temporary_paths[1],
            _records(result.action_evaluations),
            tuple(ActionEvaluation.__dataclass_fields__),
        )
        _write_csv(
            temporary_paths[2],
            _records(result.scenario_outcomes),
            tuple(ScenarioOutcome.__dataclass_fields__),
        )
        for temporary_path, target in zip(temporary_paths, targets, strict=True):
            temporary_path.replace(target)
    return targets


def execute(config_path: str | Path, *, overwrite: bool = False) -> tuple[Path, ...]:
    config = load_phase9_config(config_path)
    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    validate_weather_file(config.phase8.phase7.phase5.phase4.phase1)
    return write_artifacts(config, run_phase9(config), overwrite=overwrite)


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


def _artifact_paths(config: Phase9Config) -> tuple[Path, ...]:
    paths = (
        config.output.directory / config.output.action_events_filename,
        config.output.directory / config.output.action_evaluation_filename,
        config.output.directory / config.output.scenario_outcomes_filename,
    )
    if len(set(paths)) != len(paths):
        raise ValueError("Phase 9 output artifact file names must be unique")
    return paths


def _ensure_targets_available(targets: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to replace existing Phase 9 artifact(s): {rendered}. "
            "Pass --overwrite to replace them."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate Phase 9 treatment actions and paired outcomes."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing Phase 9 artifacts."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = execute(args.config, overwrite=args.overwrite)
    print("Phase 9 completed. Wrote:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
