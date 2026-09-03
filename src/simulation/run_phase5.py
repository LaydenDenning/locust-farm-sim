"""Command-line runner for Phase 5 sensor-only anomaly detection."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping, Sequence

from src.simulation.detection import (
    DetectionRecord,
    DetectionSurveySummary,
    Phase5Config,
    Phase5Result,
    classify_observations,
    load_phase5_config,
)
from src.simulation.observations import Phase4Result
from src.simulation.run_phase4 import run_phase4
from src.simulation.run_phase1 import validate_weather_file


def run_phase5(
    config: Phase5Config, *, phase4_result: Phase4Result | None = None
) -> Phase5Result:
    """Generate or accept Phase 4 observations, then classify them."""

    phase4_result = phase4_result or run_phase4(config.phase4)
    return classify_observations(phase4_result.observations, config.rule)


def write_artifacts(
    config: Phase5Config, result: Phase5Result, *, overwrite: bool = False
) -> tuple[Path, ...]:
    """Write Phase 5 detection records and summaries atomically."""

    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    config.output.directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase5-", dir=config.output.directory) as temporary:
        temporary_paths = tuple(Path(temporary) / target.name for target in targets)
        _write_csv(
            temporary_paths[0],
            _records(result.detections),
            tuple(DetectionRecord.__dataclass_fields__),
        )
        _write_csv(
            temporary_paths[1],
            _records(result.surveys),
            tuple(DetectionSurveySummary.__dataclass_fields__),
        )
        for temporary_path, target in zip(temporary_paths, targets, strict=True):
            temporary_path.replace(target)
    return targets


def execute(config_path: str | Path, *, overwrite: bool = False) -> tuple[Path, ...]:
    config = load_phase5_config(config_path)
    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    validate_weather_file(config.phase4.phase1)
    result = run_phase5(config)
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


def _artifact_paths(config: Phase5Config) -> tuple[Path, ...]:
    paths = (
        config.output.directory / config.output.detections_filename,
        config.output.directory / config.output.survey_summary_filename,
    )
    if len(set(paths)) != len(paths):
        raise ValueError("Phase 5 output artifact file names must be unique")
    return paths


def _ensure_targets_available(targets: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to replace existing Phase 5 artifact(s): {rendered}. "
            "Pass --overwrite to replace them."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify Phase 4 observations into Phase 5 anomaly flags."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing Phase 5 artifacts."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = execute(args.config, overwrite=args.overwrite)
    print("Phase 5 completed. Wrote:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
