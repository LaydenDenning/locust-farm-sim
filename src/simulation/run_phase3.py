"""Command-line entry point for the Phase 3 virtual drone mission."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping, Sequence

from src.farm import load_farm
from src.simulation.drone import (
    FlightLine,
    MissionPlan,
    Phase3Config,
    SortieSummary,
    ZoneCoverage,
    load_phase3_config,
    plan_mission,
)


def write_artifacts(
    config: Phase3Config, plan: MissionPlan, *, overwrite: bool = False
) -> tuple[Path, ...]:
    """Write validated Phase 3 CSV artifacts atomically."""

    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    config.output.directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".phase3-", dir=config.output.directory) as temporary:
        temporary_paths = tuple(Path(temporary) / target.name for target in targets)
        _write_csv(temporary_paths[0], _line_records(plan.lines), _line_fields())
        _write_csv(temporary_paths[1], _sortie_records(plan.sorties), _sortie_fields())
        _write_csv(
            temporary_paths[2],
            _coverage_records(plan.coverage),
            _coverage_fields(),
        )
        for temporary_path, target in zip(temporary_paths, targets, strict=True):
            temporary_path.replace(target)
    return targets


def execute(config_path: str | Path, *, overwrite: bool = False) -> tuple[Path, ...]:
    config = load_phase3_config(config_path)
    targets = _artifact_paths(config)
    _ensure_targets_available(targets, overwrite=overwrite)
    farm = load_farm(config.phase1)
    plan = plan_mission(config, farm=farm)
    return write_artifacts(config, plan, overwrite=overwrite)


def _line_records(lines: tuple[FlightLine, ...]) -> Iterable[Mapping[str, object]]:
    return (asdict(line) for line in lines)


def _sortie_records(
    sorties: tuple[SortieSummary, ...],
) -> Iterable[Mapping[str, object]]:
    return (asdict(sortie) for sortie in sorties)


def _coverage_records(
    coverage: tuple[ZoneCoverage, ...],
) -> Iterable[Mapping[str, object]]:
    for item in coverage:
        record = asdict(item)
        record["line_ids"] = "|".join(item.line_ids)
        record["first_observation_time"] = (
            item.first_observation_time.isoformat()
            if item.first_observation_time is not None
            else ""
        )
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


def _line_fields() -> tuple[str, ...]:
    return tuple(FlightLine.__dataclass_fields__)


def _sortie_fields() -> tuple[str, ...]:
    return tuple(SortieSummary.__dataclass_fields__)


def _coverage_fields() -> tuple[str, ...]:
    return tuple(ZoneCoverage.__dataclass_fields__)


def _artifact_paths(config: Phase3Config) -> tuple[Path, ...]:
    output = config.output
    paths = (
        output.directory / output.flight_lines_filename,
        output.directory / output.sortie_summary_filename,
        output.directory / output.zone_coverage_filename,
    )
    if len(set(paths)) != len(paths):
        raise ValueError("Phase 3 output artifact file names must be unique")
    return paths


def _ensure_targets_available(targets: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to replace existing Phase 3 artifact(s): {rendered}. "
            "Pass --overwrite to replace them."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan the Phase 3 drone mission.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing Phase 3 artifacts."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = execute(args.config, overwrite=args.overwrite)
    print("Phase 3 completed. Wrote:")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
