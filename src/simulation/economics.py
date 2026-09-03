"""Provisional TWSO-proxy economics for Phase 10."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.farm import Farm
from src.simulation.actions import (
    STRATEGIES,
    Phase9Config,
    Phase9Result,
    load_phase9_config,
)
from src.simulation.confirmation import Phase8Result
from src.simulation.drone import plan_mission


class EconomicsConfigError(ValueError):
    """Raised when Phase 10 configuration is invalid."""


@dataclass(frozen=True)
class EconomicAssumptions:
    currency: str
    twso_proxy_value_per_tonne: float
    flight_cost_per_sortie: float
    processing_cost_per_1000_images: float
    scout_labor_cost_per_hour: float
    confirmation_cost_per_visit: float
    false_positive_action_cost_per_ha: float

    def __post_init__(self) -> None:
        if not self.currency.strip():
            raise ValueError("economics.currency must not be empty")
        for name in (
            "twso_proxy_value_per_tonne",
            "flight_cost_per_sortie",
            "processing_cost_per_1000_images",
            "scout_labor_cost_per_hour",
            "confirmation_cost_per_visit",
            "false_positive_action_cost_per_ha",
        ):
            _nonnegative_number(getattr(self, name), f"economics.{name}")


@dataclass(frozen=True)
class Phase10OutputConfig:
    directory: Path
    zone_economics_filename: str
    strategy_summary_filename: str

    def __post_init__(self) -> None:
        for name in ("zone_economics_filename", "strategy_summary_filename"):
            value = getattr(self, name)
            if not value.strip() or Path(value).name != value:
                raise ValueError(f"output.{name} must be a simple file name")


@dataclass(frozen=True)
class Phase10Config:
    source_path: Path
    phase9: Phase9Config
    economics: EconomicAssumptions
    output: Phase10OutputConfig


@dataclass(frozen=True)
class ZoneEconomics:
    issue_id: str
    mechanism: str
    zone_id: str
    strategy: str
    zone_area_ha: float
    baseline_twso_kg_ha: float
    untreated_loss_fraction: float
    treated_loss_fraction: float
    lost_twso_kg: float
    avoided_twso_kg: float
    crop_loss_cost: float


@dataclass(frozen=True)
class StrategyEconomics:
    strategy: str
    currency: str
    affected_area_ha: float
    baseline_twso_kg: float
    lost_twso_kg: float
    avoided_twso_kg: float
    flight_cost: float
    processing_cost: float
    scouting_cost: float
    confirmation_cost: float
    treatment_cost: float
    false_positive_cost: float
    operating_cost: float
    crop_loss_cost: float
    total_cost: float
    net_benefit_vs_no_intervention: float
    break_even_drone_operations_cost: float | None


@dataclass(frozen=True)
class Phase10Result:
    zone_economics: tuple[ZoneEconomics, ...]
    strategy_summary: tuple[StrategyEconomics, ...]


def load_phase10_config(path: str | Path) -> Phase10Config:
    """Load strict Phase 10 economic assumptions."""

    source_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EconomicsConfigError(f"unable to read {source_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise EconomicsConfigError(f"invalid YAML in {source_path}: {exc}") from exc

    root = _mapping(raw, "configuration")
    _exact_keys(root, {"phase9_config", "economics", "output"}, "configuration")
    phase9 = load_phase9_config(
        _resolve_path(source_path.parent, root["phase9_config"])
    )
    values = _mapping(root["economics"], "economics")
    economic_keys = {
        "currency",
        "twso_proxy_value_per_tonne",
        "flight_cost_per_sortie",
        "processing_cost_per_1000_images",
        "scout_labor_cost_per_hour",
        "confirmation_cost_per_visit",
        "false_positive_action_cost_per_ha",
    }
    _exact_keys(values, economic_keys, "economics")
    output = _mapping(root["output"], "output")
    _exact_keys(
        output,
        {"directory", "zone_economics_filename", "strategy_summary_filename"},
        "output",
    )
    try:
        assumptions = EconomicAssumptions(
            currency=_text(values["currency"], "economics.currency"),
            twso_proxy_value_per_tonne=_number(
                values["twso_proxy_value_per_tonne"],
                "economics.twso_proxy_value_per_tonne",
            ),
            flight_cost_per_sortie=_number(
                values["flight_cost_per_sortie"],
                "economics.flight_cost_per_sortie",
            ),
            processing_cost_per_1000_images=_number(
                values["processing_cost_per_1000_images"],
                "economics.processing_cost_per_1000_images",
            ),
            scout_labor_cost_per_hour=_number(
                values["scout_labor_cost_per_hour"],
                "economics.scout_labor_cost_per_hour",
            ),
            confirmation_cost_per_visit=_number(
                values["confirmation_cost_per_visit"],
                "economics.confirmation_cost_per_visit",
            ),
            false_positive_action_cost_per_ha=_number(
                values["false_positive_action_cost_per_ha"],
                "economics.false_positive_action_cost_per_ha",
            ),
        )
        output_config = Phase10OutputConfig(
            directory=_resolve_path(source_path.parent, output["directory"]),
            zone_economics_filename=_text(
                output["zone_economics_filename"],
                "output.zone_economics_filename",
            ),
            strategy_summary_filename=_text(
                output["strategy_summary_filename"],
                "output.strategy_summary_filename",
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, EconomicsConfigError):
            raise
        raise EconomicsConfigError(str(exc)) from exc
    return Phase10Config(source_path, phase9, assumptions, output_config)


def calculate_economics(
    config: Phase10Config,
    phase8_result: Phase8Result,
    phase9_result: Phase9Result,
    *,
    farm: Farm,
    baseline_twso_kg_ha: Mapping[str, float],
) -> Phase10Result:
    """Calculate synthetic costs without changing biological outcomes."""

    _validate_baseline(farm, baseline_twso_kg_ha)
    value_per_kg = config.economics.twso_proxy_value_per_tonne / 1000.0
    zone_rows: list[ZoneEconomics] = []
    for outcome in phase9_result.scenario_outcomes:
        zone = farm.get_zone(outcome.zone_id)
        area_ha = zone.area_m2 / 10_000.0
        baseline = float(baseline_twso_kg_ha[outcome.zone_id])
        untreated_loss = baseline * area_ha * outcome.untreated_loss_fraction
        treated_loss = baseline * area_ha * outcome.treated_loss_fraction
        zone_rows.append(
            ZoneEconomics(
                issue_id=outcome.issue_id,
                mechanism=outcome.mechanism,
                zone_id=outcome.zone_id,
                strategy=outcome.strategy,
                zone_area_ha=area_ha,
                baseline_twso_kg_ha=baseline,
                untreated_loss_fraction=outcome.untreated_loss_fraction,
                treated_loss_fraction=outcome.treated_loss_fraction,
                lost_twso_kg=treated_loss,
                avoided_twso_kg=untreated_loss - treated_loss,
                crop_loss_cost=treated_loss * value_per_kg,
            )
        )

    operations = _operating_costs(config, phase8_result, phase9_result, farm)
    untreated_cost = sum(
        item.crop_loss_cost
        for item in zone_rows
        if item.strategy == "no_intervention"
    )
    summaries: list[StrategyEconomics] = []
    for strategy in STRATEGIES:
        rows = [item for item in zone_rows if item.strategy == strategy]
        baseline_total = sum(
            item.baseline_twso_kg_ha * item.zone_area_ha for item in rows
        )
        lost = sum(item.lost_twso_kg for item in rows)
        avoided = sum(item.avoided_twso_kg for item in rows)
        crop_loss = sum(item.crop_loss_cost for item in rows)
        costs = operations[strategy]
        operating = sum(costs.values())
        total = operating + crop_loss
        break_even = None
        if strategy == "drone":
            non_drone_operations = (
                costs["confirmation_cost"]
                + costs["treatment_cost"]
                + costs["false_positive_cost"]
            )
            break_even = avoided * value_per_kg - non_drone_operations
        summaries.append(
            StrategyEconomics(
                strategy=strategy,
                currency=config.economics.currency,
                affected_area_ha=sum(item.zone_area_ha for item in rows),
                baseline_twso_kg=baseline_total,
                lost_twso_kg=lost,
                avoided_twso_kg=avoided,
                flight_cost=costs["flight_cost"],
                processing_cost=costs["processing_cost"],
                scouting_cost=costs["scouting_cost"],
                confirmation_cost=costs["confirmation_cost"],
                treatment_cost=costs["treatment_cost"],
                false_positive_cost=costs["false_positive_cost"],
                operating_cost=operating,
                crop_loss_cost=crop_loss,
                total_cost=total,
                net_benefit_vs_no_intervention=untreated_cost - total,
                break_even_drone_operations_cost=break_even,
            )
        )
    return Phase10Result(tuple(zone_rows), tuple(summaries))


def _operating_costs(
    config: Phase10Config,
    phase8_result: Phase8Result,
    phase9_result: Phase9Result,
    farm: Farm,
) -> dict[str, dict[str, float]]:
    empty = {
        "flight_cost": 0.0,
        "processing_cost": 0.0,
        "scouting_cost": 0.0,
        "confirmation_cost": 0.0,
        "treatment_cost": 0.0,
        "false_positive_cost": 0.0,
    }
    costs = {strategy: dict(empty) for strategy in STRATEGIES}
    assumptions = config.economics
    phase4 = config.phase9.phase8.phase7.phase5.phase4

    sortie_count = 0
    image_count = 0
    for campaign_day in phase4.schedule.survey_days:
        mission_config = replace(
            phase4.phase3,
            survey_date=(
                phase4.phase1.calendar.base_sowing_date
                + timedelta(days=campaign_day)
            ),
            grounded=campaign_day in phase4.schedule.grounded_days,
        )
        mission = plan_mission(mission_config, farm=farm)
        sortie_count += len(mission.sorties)
        image_count += sum(item.total_image_count for item in mission.sorties)
    costs["drone"]["flight_cost"] = sortie_count * assumptions.flight_cost_per_sortie
    costs["drone"]["processing_cost"] = (
        image_count / 1000.0 * assumptions.processing_cost_per_1000_images
    )

    scout = config.phase9.phase8.phase7.phase6.scout
    completed_scout_surveys = sum(
        day not in scout.missed_days for day in phase4.schedule.survey_days
    )
    route_minutes = (
        len(scout.route_zone_ids) * scout.inspection_minutes_per_zone
        + max(0, len(scout.route_zone_ids) - 1)
        * scout.travel_minutes_between_zones
    )
    costs["scout"]["scouting_cost"] = (
        completed_scout_surveys
        * route_minutes
        / 60.0
        * assumptions.scout_labor_cost_per_hour
    )

    for method in ("drone", "scout"):
        requests = sum(
            item.method == method for item in phase8_result.confirmation_events
        )
        costs[method]["confirmation_cost"] = (
            requests * assumptions.confirmation_cost_per_visit
        )

    evaluations = {item.action_id: item for item in phase9_result.action_evaluations}
    for action in phase9_result.action_events:
        evaluation = evaluations[action.action_id]
        area_ha = farm.get_zone(action.zone_id).area_m2 / 10_000.0
        if evaluation.evaluation == "appropriate_action":
            rule = config.phase9.rule_for(evaluation.mechanism)
            costs[action.method]["treatment_cost"] += rule.cost_per_ha * area_ha
        else:
            costs[action.method]["false_positive_cost"] += (
                assumptions.false_positive_action_cost_per_ha * area_ha
            )
    return costs


def _validate_baseline(farm: Farm, values: Mapping[str, float]) -> None:
    expected = {zone.zone_id for zone in farm.zones}
    if set(values) != expected:
        raise ValueError("baseline TWSO keys must match every farm zone exactly")
    for zone_id, value in values.items():
        if not isfinite(value) or value < 0:
            raise ValueError(f"baseline TWSO for {zone_id} must be finite and nonnegative")


def _nonnegative_number(value: float, name: str) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EconomicsConfigError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise EconomicsConfigError(
            f"{name} keys do not match schema; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _resolve_path(parent: Path, value: Any) -> Path:
    path = Path(_text(value, "path"))
    return (parent / path).resolve() if not path.is_absolute() else path.resolve()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EconomicsConfigError(f"{name} must be nonempty text")
    return value.strip()


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise EconomicsConfigError(f"{name} must be numeric")
    return float(value)
