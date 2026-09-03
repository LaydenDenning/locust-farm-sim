"""Command-line runner for Phase 8 human confirmation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping, Sequence

from src.simulation.confirmation import (
    ConfirmationEvaluation,
    ConfirmationEvent,
    IssueConfirmationSummary,
    Phase8Config,
    Phase8Result,
    load_phase8_config,
    simulate_confirmations,
)
from src.simulation.run_phase1 import validate_weather_file
from src.simulation.run_phase7 import run_phase7


def run_phase8(config: Phase8Config) -> Phase8Result:
    """Run Phase 7 and simulate confirmation of all flags."""

    return simulate_confirmations(config, run_phase7(config.phase7))


def write_artifacts(
    config: Phase8Config, result: Phase8Result, *, overwrite: bool = False
) -> tuple[Path, ...]:
    """Write Phase 8 operational and evaluation artifacts atomically."""

    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    config.output.directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase8-", dir=config.output.directory) as temporary:
        temporary_paths = tuple(Path(temporary) / target.name for target in targets)
        _write_csv(
            temporary_paths[0],
            _records(result.confirmation_events),
            tuple(ConfirmationEvent.__dataclass_fields__),
        )
        _write_csv(
            temporary_paths[1],
            _records(result.evaluations),
            tuple(ConfirmationEvaluation.__dataclass_fields__),
        )
        _write_csv(
            temporary_paths[2],
            _records(result.issue_summaries),
            tuple(IssueConfirmationSummary.__dataclass_fields__),
        )
        for temporary_path, target in zip(temporary_paths, targets, strict=True):
            temporary_path.replace(target)
    return targets


def execute(config_path: str | Path, *, overwrite: bool = False) -> tuple[Path, ...]:
    config = load_phase8_config(config_path)
    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    validate_weather_file(config.phase7.phase5.phase4.phase1)
    return write_artifacts(config, run_phase8(config), overwrite=overwrite)


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


def _artifact_paths(config: Phase8Config) -> tuple[Path, ...]:
    paths = (
        config.output.directory / config.output.confirmation_events_filename,
        config.output.directory / config.output.confirmation_evaluation_filename,
        config.output.directory / config.output.issue_summary_filename,
    )
    if len(set(paths)) != len(paths):
        raise ValueError("Phase 8 output artifact file names must be unique")
    return paths


def _ensure_targets_available(targets: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to replace existing Phase 8 artifact(s): {rendered}. "
            "Pass --overwrite to replace them."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate Phase 8 human confirmation of anomaly flags."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing Phase 8 artifacts."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = execute(args.config, overwrite=args.overwrite)
    print("Phase 8 completed. Wrote:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
