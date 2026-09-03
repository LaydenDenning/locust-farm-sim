"""Command-line runner for Phase 4 zone-level RGB/NIR observations."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping, Sequence

from src.crop import TruthModel
from src.farm import Farm, load_farm
from src.simulation.observations import (
    Observation,
    Phase4Config,
    Phase4Result,
    SurveySummary,
    load_phase4_config,
    simulate_observations,
)
from src.simulation.run_phase1 import run_farm, validate_weather_file


def run_phase4(
    config: Phase4Config,
    *,
    farm: Farm | None = None,
    truth_model: TruthModel | None = None,
) -> Phase4Result:
    """Run crop truth once, then generate all configured observations."""

    farm = farm or load_farm(config.phase1)
    truth_model = truth_model or TruthModel(config.phase1)
    truth = run_farm(config.phase1, farm=farm, truth_model=truth_model)
    return simulate_observations(config, truth.daily, farm=farm)


def write_artifacts(
    config: Phase4Config, result: Phase4Result, *, overwrite: bool = False
) -> tuple[Path, ...]:
    """Write Phase 4 observations and survey summaries atomically."""

    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    config.output.directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase4-", dir=config.output.directory) as temporary:
        temporary_paths = tuple(Path(temporary) / target.name for target in targets)
        _write_csv(
            temporary_paths[0],
            _records(result.observations),
            tuple(Observation.__dataclass_fields__),
        )
        _write_csv(
            temporary_paths[1],
            _records(result.surveys),
            tuple(SurveySummary.__dataclass_fields__),
        )
        for temporary_path, target in zip(temporary_paths, targets, strict=True):
            temporary_path.replace(target)
    return targets


def execute(config_path: str | Path, *, overwrite: bool = False) -> tuple[Path, ...]:
    config = load_phase4_config(config_path)
    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    validate_weather_file(config.phase1)
    result = run_phase4(config)
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


def _artifact_paths(config: Phase4Config) -> tuple[Path, ...]:
    output = config.output
    paths = (
        output.directory / output.observations_filename,
        output.directory / output.survey_summary_filename,
    )
    if len(set(paths)) != len(paths):
        raise ValueError("Phase 4 output artifact file names must be unique")
    return paths


def _ensure_targets_available(targets: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to replace existing Phase 4 artifact(s): {rendered}. "
            "Pass --overwrite to replace them."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Phase 4 zone-level RGB/NIR observations."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing Phase 4 artifacts."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = execute(args.config, overwrite=args.overwrite)
    print("Phase 4 completed. Wrote:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
