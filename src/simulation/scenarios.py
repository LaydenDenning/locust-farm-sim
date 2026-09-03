"""Seeded paired scenario experiments for Phase 11."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite, sqrt
from pathlib import Path
from random import Random
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from src.farm import Farm
from src.simulation.actions import simulate_actions
from src.simulation.comparison import compare_methods
from src.simulation.confirmation import simulate_confirmations
from src.simulation.detection import classify_observations
from src.simulation.economics import (
    Phase10Config,
    calculate_economics,
    load_phase10_config,
)
from src.simulation.issues import ISSUE_MECHANISMS, IssueScenario
from src.simulation.observations import simulate_observations
from src.simulation.scouting import simulate_scouting


class ScenarioConfigError(ValueError):
    """Raised when Phase 11 configuration is invalid."""


@dataclass(frozen=True)
class NumberRange:
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not isfinite(self.minimum) or not isfinite(self.maximum):
            raise ValueError("range bounds must be finite")
        if self.minimum > self.maximum:
            raise ValueError("range minimum must not exceed maximum")


@dataclass(frozen=True)
class IntegerRange:
    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum < 0 or self.minimum > self.maximum:
            raise ValueError("integer range must be nonnegative and ordered")


@dataclass(frozen=True)
class ScenarioSettings:
    count: int
    seed: int
    no_issue_probability: float
    footprint_zone_count: IntegerRange
    onset_day: IntegerRange
    progression_per_day: NumberRange
    max_severity: NumberRange
    visibility_delay_days: IntegerRange
    visibility_scale: NumberRange
    untreated_loss_fraction: NumberRange
    drone_grounding_probability: float
    scout_missed_probability: float
    efficacy_multiplier: NumberRange
    response_delay_days: IntegerRange
    treatment_cost_multiplier: NumberRange
    twso_value_multiplier: NumberRange

    def __post_init__(self) -> None:
        if not isinstance(self.count, int) or isinstance(self.count, bool) or self.count < 1:
            raise ValueError("scenarios.count must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("scenarios.seed must be an integer")
        for name in (
            "no_issue_probability",
            "drone_grounding_probability",
            "scout_missed_probability",
        ):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"scenarios.{name} must be between 0 and 1")
        for name in (
            "progression_per_day",
            "max_severity",
            "visibility_scale",
            "untreated_loss_fraction",
            "efficacy_multiplier",
            "treatment_cost_multiplier",
            "twso_value_multiplier",
        ):
            value = getattr(self, name)
            if value.minimum <= 0:
                raise ValueError(f"scenarios.{name} must be positive")
        for name in ("max_severity", "visibility_scale", "untreated_loss_fraction"):
            if getattr(self, name).maximum > 1:
                raise ValueError(f"scenarios.{name} must not exceed 1")


@dataclass(frozen=True)
class Phase11OutputConfig:
    directory: Path
    scenario_results_filename: str
    distribution_summary_filename: str
    sensitivity_filename: str

    def __post_init__(self) -> None:
        for name in (
            "scenario_results_filename",
            "distribution_summary_filename",
            "sensitivity_filename",
        ):
            value = getattr(self, name)
            if not value.strip() or Path(value).name != value:
                raise ValueError(f"output.{name} must be a simple file name")


@dataclass(frozen=True)
class Phase11Config:
    source_path: Path
    phase10: Phase10Config
    scenarios: ScenarioSettings
    output: Phase11OutputConfig


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    scenario_seed: int
    mechanism: str
    zone_ids: str
    footprint_zone_count: int
    onset_day: int
    progression_per_day: float
    max_severity: float
    visibility_delay_days: int
    visibility_scale: float
    untreated_loss_fraction: float
    drone_grounded_surveys: int
    scout_missed_surveys: int
    response_delay_days: int
    efficacy: float
    treatment_cost_per_ha: float
    twso_proxy_value_per_tonne: float
    drone_avoided_twso_kg: float
    scout_avoided_twso_kg: float
    drone_net_benefit: float
    scout_net_benefit: float
    drone_advantage_vs_scout: float


@dataclass(frozen=True)
class DistributionSummary:
    metric: str
    sample_count: int
    median: float
    percentile_05: float
    percentile_95: float
    probability_above_zero: float


@dataclass(frozen=True)
class SensitivityResult:
    parameter: str
    correlation_with_drone_advantage: float


@dataclass(frozen=True)
class Phase11Result:
    scenarios: tuple[ScenarioResult, ...]
    distributions: tuple[DistributionSummary, ...]
    sensitivities: tuple[SensitivityResult, ...]


def load_phase11_config(path: str | Path) -> Phase11Config:
    """Load strict Phase 11 scenario ranges."""

    source_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ScenarioConfigError(f"unable to read {source_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ScenarioConfigError(f"invalid YAML in {source_path}: {exc}") from exc
    root = _mapping(raw, "configuration")
    _exact_keys(root, {"phase10_config", "scenarios", "output"}, "configuration")
    phase10 = load_phase10_config(
        _resolve_path(source_path.parent, root["phase10_config"])
    )
    values = _mapping(root["scenarios"], "scenarios")
    expected = {
        "count", "seed", "no_issue_probability", "footprint_zone_count",
        "onset_day", "progression_per_day", "max_severity",
        "visibility_delay_days", "visibility_scale", "untreated_loss_fraction",
        "drone_grounding_probability", "scout_missed_probability",
        "efficacy_multiplier", "response_delay_days", "treatment_cost_multiplier",
        "twso_value_multiplier",
    }
    _exact_keys(values, expected, "scenarios")
    output = _mapping(root["output"], "output")
    _exact_keys(
        output,
        {"directory", "scenario_results_filename", "distribution_summary_filename", "sensitivity_filename"},
        "output",
    )
    try:
        settings = ScenarioSettings(
            count=_integer(values["count"], "scenarios.count"),
            seed=_integer(values["seed"], "scenarios.seed"),
            no_issue_probability=_number(values["no_issue_probability"], "scenarios.no_issue_probability"),
            footprint_zone_count=_integer_range(values["footprint_zone_count"], "scenarios.footprint_zone_count"),
            onset_day=_integer_range(values["onset_day"], "scenarios.onset_day"),
            progression_per_day=_number_range(values["progression_per_day"], "scenarios.progression_per_day"),
            max_severity=_number_range(values["max_severity"], "scenarios.max_severity"),
            visibility_delay_days=_integer_range(values["visibility_delay_days"], "scenarios.visibility_delay_days"),
            visibility_scale=_number_range(values["visibility_scale"], "scenarios.visibility_scale"),
            untreated_loss_fraction=_number_range(values["untreated_loss_fraction"], "scenarios.untreated_loss_fraction"),
            drone_grounding_probability=_number(values["drone_grounding_probability"], "scenarios.drone_grounding_probability"),
            scout_missed_probability=_number(values["scout_missed_probability"], "scenarios.scout_missed_probability"),
            efficacy_multiplier=_number_range(values["efficacy_multiplier"], "scenarios.efficacy_multiplier"),
            response_delay_days=_integer_range(values["response_delay_days"], "scenarios.response_delay_days"),
            treatment_cost_multiplier=_number_range(values["treatment_cost_multiplier"], "scenarios.treatment_cost_multiplier"),
            twso_value_multiplier=_number_range(values["twso_value_multiplier"], "scenarios.twso_value_multiplier"),
        )
        phase1 = phase10.phase9.phase8.phase7.phase5.phase4.phase1
        zone_count = phase1.field.rows * phase1.field.columns
        if settings.footprint_zone_count.maximum > zone_count:
            raise ValueError("footprint range exceeds farm zone count")
        if settings.onset_day.maximum >= phase1.calendar.max_duration_days:
            raise ValueError("onset range must fit within the crop campaign")
        output_config = Phase11OutputConfig(
            directory=_resolve_path(source_path.parent, output["directory"]),
            scenario_results_filename=_text(output["scenario_results_filename"], "output.scenario_results_filename"),
            distribution_summary_filename=_text(output["distribution_summary_filename"], "output.distribution_summary_filename"),
            sensitivity_filename=_text(output["sensitivity_filename"], "output.sensitivity_filename"),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ScenarioConfigError):
            raise
        raise ScenarioConfigError(str(exc)) from exc
    return Phase11Config(source_path, phase10, settings, output_config)


def run_scenarios(
    config: Phase11Config,
    daily_truth: pd.DataFrame,
    *,
    farm: Farm,
    baseline_twso_kg_ha: Mapping[str, float],
) -> Phase11Result:
    """Run paired drone/scout scenarios against one fixed crop baseline."""

    master = Random(config.scenarios.seed)
    rows: list[ScenarioResult] = []
    for number in range(1, config.scenarios.count + 1):
        scenario_seed = master.randrange(0, 2**32)
        rows.append(
            _run_one(
                config,
                scenario_id=f"SCENARIO_{number:04d}",
                scenario_seed=scenario_seed,
                daily_truth=daily_truth,
                farm=farm,
                baseline_twso_kg_ha=baseline_twso_kg_ha,
            )
        )
    scenarios = tuple(rows)
    return Phase11Result(
        scenarios,
        _distribution_summaries(scenarios),
        _sensitivity_results(scenarios),
    )


def _run_one(
    config: Phase11Config,
    *,
    scenario_id: str,
    scenario_seed: int,
    daily_truth: pd.DataFrame,
    farm: Farm,
    baseline_twso_kg_ha: Mapping[str, float],
) -> ScenarioResult:
    random = Random(scenario_seed)
    settings = config.scenarios
    base10 = config.phase10
    base4 = base10.phase9.phase8.phase7.phase5.phase4
    issue, mechanism, zone_ids = _sample_issue(random, settings, farm, scenario_id)
    survey_days = base4.schedule.survey_days
    grounded = tuple(day for day in survey_days if random.random() < settings.drone_grounding_probability)
    missed = tuple(day for day in survey_days if random.random() < settings.scout_missed_probability)

    phase4 = replace(
        base4,
        issues=() if issue is None else (issue,),
        schedule=replace(base4.schedule, grounded_days=grounded),
        noise=replace(base4.noise, seed=random.randrange(0, 2**32)),
    )
    base7 = base10.phase9.phase8.phase7
    phase5 = replace(base7.phase5, phase4=phase4)
    phase6 = replace(
        base7.phase6,
        phase4=phase4,
        scout=replace(base7.phase6.scout, seed=random.randrange(0, 2**32), missed_days=missed),
    )
    phase7 = replace(base7, phase5=phase5, phase6=phase6)
    base8 = base10.phase9.phase8
    phase8_config = replace(
        base8,
        phase7=phase7,
        confirmation=replace(base8.confirmation, seed=random.randrange(0, 2**32)),
    )

    efficacy = 0.0
    response_delay = 0
    treatment_cost = 0.0
    rules = list(base10.phase9.intervention_rules)
    if issue is not None:
        multiplier = _sample_number(random, settings.efficacy_multiplier)
        response_delay = _sample_integer(random, settings.response_delay_days)
        cost_multiplier = _sample_number(random, settings.treatment_cost_multiplier)
        for index, rule in enumerate(rules):
            if rule.mechanism == issue.mechanism:
                efficacy = min(1.0, rule.efficacy * multiplier)
                treatment_cost = rule.cost_per_ha * cost_multiplier
                rules[index] = replace(
                    rule,
                    response_delay_days=response_delay,
                    efficacy=efficacy,
                    cost_per_ha=treatment_cost,
                )
                break
    phase9_config = replace(base10.phase9, phase8=phase8_config, intervention_rules=tuple(rules))
    value_multiplier = _sample_number(random, settings.twso_value_multiplier)
    phase10_config = replace(
        base10,
        phase9=phase9_config,
        economics=replace(
            base10.economics,
            twso_proxy_value_per_tonne=(
                base10.economics.twso_proxy_value_per_tonne * value_multiplier
            ),
        ),
    )

    phase4_result = simulate_observations(phase4, daily_truth, farm=farm)
    drone = classify_observations(phase4_result.observations, phase5.rule)
    scout = simulate_scouting(phase6, farm=farm)
    phase7_result = compare_methods(phase7, drone, scout)
    phase8_result = simulate_confirmations(phase8_config, phase7_result)
    phase9_result = simulate_actions(phase9_config, phase8_result)
    phase10_result = calculate_economics(
        phase10_config,
        phase8_result,
        phase9_result,
        farm=farm,
        baseline_twso_kg_ha=baseline_twso_kg_ha,
    )
    economics = {item.strategy: item for item in phase10_result.strategy_summary}
    drone_result = economics["drone"]
    scout_result = economics["scout"]
    return ScenarioResult(
        scenario_id=scenario_id,
        scenario_seed=scenario_seed,
        mechanism=mechanism,
        zone_ids="|".join(zone_ids),
        footprint_zone_count=len(zone_ids),
        onset_day=issue.onset_day if issue else 0,
        progression_per_day=issue.progression_per_day if issue else 0.0,
        max_severity=issue.max_severity if issue else 0.0,
        visibility_delay_days=issue.visibility_delay_days if issue else 0,
        visibility_scale=issue.visibility_scale if issue else 0.0,
        untreated_loss_fraction=issue.untreated_loss_fraction if issue else 0.0,
        drone_grounded_surveys=len(grounded),
        scout_missed_surveys=len(missed),
        response_delay_days=response_delay,
        efficacy=efficacy,
        treatment_cost_per_ha=treatment_cost,
        twso_proxy_value_per_tonne=phase10_config.economics.twso_proxy_value_per_tonne,
        drone_avoided_twso_kg=drone_result.avoided_twso_kg,
        scout_avoided_twso_kg=scout_result.avoided_twso_kg,
        drone_net_benefit=drone_result.net_benefit_vs_no_intervention,
        scout_net_benefit=scout_result.net_benefit_vs_no_intervention,
        drone_advantage_vs_scout=(
            drone_result.net_benefit_vs_no_intervention
            - scout_result.net_benefit_vs_no_intervention
        ),
    )


def _sample_issue(
    random: Random,
    settings: ScenarioSettings,
    farm: Farm,
    scenario_id: str,
) -> tuple[IssueScenario | None, str, tuple[str, ...]]:
    if random.random() < settings.no_issue_probability:
        return None, "none", ()
    mechanism = random.choice(ISSUE_MECHANISMS)
    count = _sample_integer(random, settings.footprint_zone_count)
    zone_ids = _contiguous_footprint(random, farm, count)
    issue = IssueScenario(
        issue_id=f"{scenario_id}_ISSUE",
        mechanism=mechanism,
        zone_ids=zone_ids,
        onset_day=_sample_integer(random, settings.onset_day),
        progression_per_day=_sample_number(random, settings.progression_per_day),
        max_severity=_sample_number(random, settings.max_severity),
        visibility_delay_days=_sample_integer(random, settings.visibility_delay_days),
        visibility_scale=_sample_number(random, settings.visibility_scale),
        untreated_loss_fraction=_sample_number(random, settings.untreated_loss_fraction),
    )
    return issue, mechanism, zone_ids


def _contiguous_footprint(random: Random, farm: Farm, count: int) -> tuple[str, ...]:
    by_position = {(zone.row, zone.column): zone for zone in farm.zones}
    first = random.choice(tuple(sorted(farm.zones, key=lambda item: (item.row, item.column))))
    selected = {(first.row, first.column)}
    while len(selected) < count:
        candidates = set()
        for row, column in selected:
            for position in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                if position in by_position and position not in selected:
                    candidates.add(position)
        selected.add(random.choice(tuple(sorted(candidates))))
    return tuple(by_position[position].zone_id for position in sorted(selected))


def _distribution_summaries(rows: Sequence[ScenarioResult]) -> tuple[DistributionSummary, ...]:
    metrics = {
        "drone_net_benefit": [item.drone_net_benefit for item in rows],
        "scout_net_benefit": [item.scout_net_benefit for item in rows],
        "drone_advantage_vs_scout": [item.drone_advantage_vs_scout for item in rows],
        "drone_avoided_twso_kg": [item.drone_avoided_twso_kg for item in rows],
        "scout_avoided_twso_kg": [item.scout_avoided_twso_kg for item in rows],
    }
    return tuple(
        DistributionSummary(
            metric=name,
            sample_count=len(values),
            median=_percentile(values, 0.50),
            percentile_05=_percentile(values, 0.05),
            percentile_95=_percentile(values, 0.95),
            probability_above_zero=sum(value > 0 for value in values) / len(values),
        )
        for name, values in metrics.items()
    )


def _sensitivity_results(rows: Sequence[ScenarioResult]) -> tuple[SensitivityResult, ...]:
    parameters = (
        "footprint_zone_count", "onset_day", "progression_per_day", "max_severity",
        "visibility_delay_days", "visibility_scale", "untreated_loss_fraction",
        "drone_grounded_surveys", "scout_missed_surveys", "response_delay_days",
        "efficacy", "treatment_cost_per_ha", "twso_proxy_value_per_tonne",
    )
    target = [item.drone_advantage_vs_scout for item in rows]
    return tuple(
        SensitivityResult(
            parameter=name,
            correlation_with_drone_advantage=_correlation(
                [float(getattr(item, name)) for item in rows], target
            ),
        )
        for name in parameters
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _correlation(first: Sequence[float], second: Sequence[float]) -> float:
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    first_delta = [value - first_mean for value in first]
    second_delta = [value - second_mean for value in second]
    denominator = sqrt(
        sum(value * value for value in first_delta)
        * sum(value * value for value in second_delta)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(first_delta, second_delta, strict=True)) / denominator


def _sample_number(random: Random, value: NumberRange) -> float:
    return random.uniform(value.minimum, value.maximum)


def _sample_integer(random: Random, value: IntegerRange) -> int:
    return random.randint(value.minimum, value.maximum)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScenarioConfigError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise ScenarioConfigError(f"{name} keys do not match schema; missing={sorted(missing)}, extra={sorted(extra)}")


def _resolve_path(parent: Path, value: Any) -> Path:
    path = Path(_text(value, "path"))
    return (parent / path).resolve() if not path.is_absolute() else path.resolve()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScenarioConfigError(f"{name} must be nonempty text")
    return value.strip()


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ScenarioConfigError(f"{name} must be an integer")
    return value


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ScenarioConfigError(f"{name} must be numeric")
    return float(value)


def _range_values(value: Any, name: str) -> tuple[Any, Any]:
    if not isinstance(value, list) or len(value) != 2:
        raise ScenarioConfigError(f"{name} must contain [minimum, maximum]")
    return value[0], value[1]


def _number_range(value: Any, name: str) -> NumberRange:
    minimum, maximum = _range_values(value, name)
    return NumberRange(_number(minimum, name), _number(maximum, name))


def _integer_range(value: Any, name: str) -> IntegerRange:
    minimum, maximum = _range_values(value, name)
    return IntegerRange(_integer(minimum, name), _integer(maximum, name))
