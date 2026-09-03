"""Command-line runner for Phase 7 detection comparison."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping, Sequence

from src.simulation.comparison import (
    DetectionEvaluation,
    IssueDetectionSummary,
    MethodDetection,
    Phase7Config,
    Phase7Result,
    compare_methods,
    load_phase7_config,
)
from src.simulation.run_phase1 import validate_weather_file
from src.simulation.run_phase5 import run_phase5
from src.simulation.scouting import simulate_scouting


def run_phase7(config: Phase7Config) -> Phase7Result:
    """Run both observation methods, then compare their detections."""

    drone_result = run_phase5(config.phase5)
    scout_result = simulate_scouting(config.phase6)
    return compare_methods(config, drone_result, scout_result)


def write_artifacts(
    config: Phase7Config, result: Phase7Result, *, overwrite: bool = False
) -> tuple[Path, ...]:
    """Write Phase 7 comparison artifacts atomically."""

    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    config.output.directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase7-", dir=config.output.directory) as temporary:
        temporary_paths = tuple(Path(temporary) / target.name for target in targets)
        _write_csv(
            temporary_paths[0],
            _records(result.method_detections),
            tuple(MethodDetection.__dataclass_fields__),
        )
        _write_csv(
            temporary_paths[1],
            _records(result.evaluations),
            tuple(DetectionEvaluation.__dataclass_fields__),
        )
        _write_csv(
            temporary_paths[2],
            _records(result.issue_summaries),
            tuple(IssueDetectionSummary.__dataclass_fields__),
        )
        for temporary_path, target in zip(temporary_paths, targets, strict=True):
            temporary_path.replace(target)
    return targets


def execute(config_path: str | Path, *, overwrite: bool = False) -> tuple[Path, ...]:
    config = load_phase7_config(config_path)
    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    validate_weather_file(config.phase5.phase4.phase1)
    result = run_phase7(config)
    return write_artifacts(config, result, overwrite=overwrite)


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


def _artifact_paths(config: Phase7Config) -> tuple[Path, ...]:
    paths = (
        config.output.directory / config.output.method_detections_filename,
        config.output.directory / config.output.detection_evaluation_filename,
        config.output.directory / config.output.issue_summary_filename,
    )
    if len(set(paths)) != len(paths):
        raise ValueError("Phase 7 output artifact file names must be unique")
    return paths


def _ensure_targets_available(targets: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to replace existing Phase 7 artifact(s): {rendered}. "
            "Pass --overwrite to replace them."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Phase 7 drone and scout detections."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing Phase 7 artifacts."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = execute(args.config, overwrite=args.overwrite)
    print("Phase 7 completed. Wrote:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
