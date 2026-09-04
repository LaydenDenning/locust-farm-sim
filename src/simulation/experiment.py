"""Paired, method-agnostic crop-monitoring experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from math import sqrt
from random import Random
from statistics import mean, median
from typing import Mapping, Sequence

import pandas as pd

from src.farm import Farm
from src.simulation.issues import ISSUE_MECHANISMS, IssueScenario, evaluate_intervention
from src.simulation.monitoring import CanonicalDetection, MonitoringResult, run_method
from src.simulation.profiles import ExperimentConfig, IntegerRange, NumberRange
from src.simulation.spatial import AnalysisCell, build_analysis_grid, contiguous_cells, zone_fractions


@dataclass(frozen=True)
class ScenarioDraw:
    scenario_id: str
    scenario_seed: int
    issue: IssueScenario | None
    affected_fractions: Mapping[str, float]
    response_delay_days: int
    efficacy_multiplier: float
    treatment_cost_multiplier: float
    twso_value_multiplier: float


@dataclass(frozen=True)
class ScenarioMethodResult:
    scenario_id: str
    scenario_seed: int
    method_id: str
    method_kind: str
    mechanism: str
    zone_ids: str
    affected_area_ha: float
    scheduled_surveys: int
    completed_surveys: int
    unavailable_observations: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    issue_detected: bool
    first_detection_delay_days: int | None
    first_confirmation_delay_days: int | None
    first_action_delay_days: int | None
    avoided_twso_kg: float
    operating_cost: float
    confirmation_cost: float
    treatment_cost: float
    false_positive_cost: float
    compatibility_adjustment_cost: float
    crop_loss_cost: float
    total_cost: float
    net_benefit_vs_no_intervention: float


@dataclass(frozen=True)
class PairwiseResult:
    scenario_id: str
    candidate_method_id: str
    reference_method_id: str
    detection_delay_delta_days: float | None
    avoided_twso_delta_kg: float
    total_cost_delta: float
    net_benefit_delta: float
    candidate_financially_better: bool


@dataclass(frozen=True)
class DistributionSummary:
    scope: str
    result_id: str
    metric: str
    sample_count: int
    mean: float
    median: float
    percentile_05: float
    percentile_95: float
    probability_above_zero: float


@dataclass(frozen=True)
class SensitivityResult:
    scope: str
    result_id: str
    target_metric: str
    parameter: str
    correlation: float


@dataclass(frozen=True)
class ExperimentResult:
    method_results: tuple[ScenarioMethodResult, ...]
    pairwise_results: tuple[PairwiseResult, ...]
    distributions: tuple[DistributionSummary, ...]
    sensitivities: tuple[SensitivityResult, ...]


def run_experiment(
    config: ExperimentConfig,
    daily_truth: pd.DataFrame,
    *,
    farm: Farm,
    baseline_twso_kg_ha: Mapping[str, float],
) -> ExperimentResult:
    """Evaluate every configured method against identical scenario draws."""

    missing = set(config.crop_profile.required_truth_variables) - set(daily_truth.columns)
    if missing:
        raise ValueError(f"daily truth is missing crop-profile variables: {sorted(missing)}")
    expected_zones = {zone.zone_id for zone in farm.zones}
    if set(baseline_twso_kg_ha) != expected_zones:
        raise ValueError("baseline TWSO keys must match farm zones")
    cells = build_analysis_grid(farm, config.scenario_profile.analysis_cell_size_m)
    master = Random(config.seed)
    rows: list[ScenarioMethodResult] = []
    parameters: dict[str, Mapping[str, float]] = {}
    for number in range(1, config.scenario_count + 1):
        seed = master.randrange(0, 2**32)
        draw = _draw_scenario(config, farm, cells, f"SCENARIO_{number:04d}", seed)
        parameters[draw.scenario_id] = _scenario_parameters(draw)
        for method in config.methods:
            monitoring = run_method(method, farm=farm, cells=cells, crop=config.crop_profile, issue=draw.issue, affected_fractions=draw.affected_fractions, scenario_seed=seed)
            rows.append(_evaluate_method(config, draw, monitoring, farm, baseline_twso_kg_ha))
    method_results = _apply_legacy_compatibility(config, tuple(rows), daily_truth, farm, baseline_twso_kg_ha)
    pairwise = _pairwise(method_results, config.reference_method_id)
    return ExperimentResult(method_results, pairwise, _distributions(method_results, pairwise), _sensitivities(method_results, pairwise, parameters))


def _draw_scenario(config: ExperimentConfig, farm: Farm, cells: tuple[AnalysisCell, ...], scenario_id: str, seed: int) -> ScenarioDraw:
    random = Random(seed)
    settings = config.scenario_profile
    if random.random() < settings.no_issue_probability:
        issue = None
        fractions = {zone.zone_id: 0.0 for zone in farm.zones}
    elif settings.spatial_footprints:
        count = _sample_integer(random, settings.footprint_cell_count)
        start = random.choice(cells).cell_id
        selected = contiguous_cells(cells, start, count)
        fractions = dict(zone_fractions(cells, selected))
        zone_ids = tuple(zone_id for zone_id, fraction in fractions.items() if fraction > 0)
        issue = _issue(random, scenario_id, zone_ids, settings)
    else:
        mechanism = random.choice(ISSUE_MECHANISMS)
        count = _sample_integer(random, settings.footprint_zone_count)
        zone_ids = _contiguous_zones(random, farm, count)
        fractions = {zone.zone_id: (1.0 if zone.zone_id in zone_ids else 0.0) for zone in farm.zones}
        issue = _issue(random, scenario_id, zone_ids, settings, mechanism=mechanism)
    # Preserve the established Phase 11 draw order for comparable aggregate
    # scenario inputs. Method failures themselves use keyed streams below.
    for _ in range(20):
        random.random()
    for _ in range(3):
        random.randrange(0, 2**32)
    if issue is None:
        response_delay = 0
        efficacy_multiplier = 0.0
        treatment_cost_multiplier = 0.0
    else:
        efficacy_multiplier = _sample_number(random, settings.efficacy_multiplier)
        response_delay = _sample_integer(random, settings.response_delay_days)
        treatment_cost_multiplier = _sample_number(random, settings.treatment_cost_multiplier)
    return ScenarioDraw(scenario_id, seed, issue, fractions, response_delay, efficacy_multiplier, treatment_cost_multiplier, _sample_number(random, settings.twso_value_multiplier))


def _issue(random: Random, scenario_id: str, zone_ids: tuple[str, ...], settings: object, *, mechanism: str | None = None) -> IssueScenario:
    return IssueScenario(
        issue_id=f"{scenario_id}_ISSUE",
        mechanism=mechanism or random.choice(ISSUE_MECHANISMS),
        zone_ids=zone_ids,
        onset_day=_sample_integer(random, settings.onset_day),
        progression_per_day=_sample_number(random, settings.progression_per_day),
        max_severity=_sample_number(random, settings.max_severity),
        visibility_delay_days=_sample_integer(random, settings.visibility_delay_days),
        visibility_scale=_sample_number(random, settings.visibility_scale),
        untreated_loss_fraction=_sample_number(random, settings.untreated_loss_fraction),
    )


def _evaluate_method(config: ExperimentConfig, draw: ScenarioDraw, monitoring: MonitoringResult, farm: Farm, baseline: Mapping[str, float]) -> ScenarioMethodResult:
    issue = draw.issue
    onset = config.phase1.calendar.base_sowing_date + timedelta(days=issue.onset_day) if issue else None
    counts = {name: 0 for name in ("true_positives", "false_positives", "false_negatives", "true_negatives")}
    true_flags: list[CanonicalDetection] = []
    confirmed_true: list[CanonicalDetection] = []
    confirmed_false: list[CanonicalDetection] = []
    sensitivity = float(config.economics["confirmation_sensitivity"])
    specificity = float(config.economics["confirmation_specificity"])
    for item in monitoring.detections:
        truth = issue is not None and item.zone_id in issue.zone_ids and item.survey_date >= onset
        if item.status == "unavailable":
            continue
        if truth and item.status == "flagged":
            counts["true_positives"] += 1
            true_flags.append(item)
        elif truth:
            counts["false_negatives"] += 1
        elif item.status == "flagged":
            counts["false_positives"] += 1
        else:
            counts["true_negatives"] += 1
        if item.status == "flagged":
            probability = sensitivity if truth else 1.0 - specificity
            if Random(_keyed_seed(draw.scenario_seed, monitoring.method_id, item.survey_id, item.zone_id, "confirmation")).random() < probability:
                (confirmed_true if truth else confirmed_false).append(item)

    confirmation_cost = (len(true_flags) + counts["false_positives"]) * float(config.economics["confirmation_cost_per_visit"])
    false_positive_cost = 0.0
    for item in confirmed_false:
        false_positive_cost += farm.get_zone(item.zone_id).area_m2 / 10_000.0 * float(config.economics["false_positive_action_cost_per_ha"])
    untreated_crop_cost = 0.0
    crop_loss_cost = 0.0
    avoided_twso = 0.0
    treatment_cost = 0.0
    action_dates = []
    if issue is not None:
        base_rule = next(rule for rule in config.intervention_rules if rule.mechanism == issue.mechanism)
        rule = replace(base_rule, response_delay_days=draw.response_delay_days, efficacy=min(1.0, base_rule.efficacy * draw.efficacy_multiplier), cost_per_ha=base_rule.cost_per_ha * draw.treatment_cost_multiplier)
        value_per_kg = config.crop_profile.twso_proxy_value_per_tonne * draw.twso_value_multiplier / 1000.0
        for zone_id in issue.zone_ids:
            zone = farm.get_zone(zone_id)
            affected_area_ha = zone.area_m2 / 10_000.0 * draw.affected_fractions[zone_id]
            untreated = baseline[zone_id] * affected_area_ha * issue.untreated_loss_fraction
            untreated_crop_cost += untreated * value_per_kg
            candidates = [item for item in confirmed_true if item.zone_id == zone_id]
            if candidates:
                first = min(candidates, key=lambda item: (item.available_date, item.survey_id))
                action_date = first.available_date + timedelta(days=next(method.confirmation_delay_days for method in config.methods if method.method_id == monitoring.method_id))
                action_day = (action_date - config.phase1.calendar.base_sowing_date).days
                outcome = evaluate_intervention(issue, campaign_days=config.phase1.calendar.max_duration_days, action_day=action_day, rule=rule)
                treatment_cost += affected_area_ha * outcome.action_cost_per_ha
                action_dates.append(action_date)
            else:
                outcome = evaluate_intervention(issue, campaign_days=config.phase1.calendar.max_duration_days)
            lost = baseline[zone_id] * affected_area_ha * outcome.treated_loss_fraction
            crop_loss_cost += lost * value_per_kg
            avoided_twso += baseline[zone_id] * affected_area_ha * outcome.avoided_loss_fraction
    total = monitoring.operating_cost + confirmation_cost + treatment_cost + false_positive_cost + crop_loss_cost
    first_detection = min((item.available_date for item in true_flags), default=None)
    first_confirmation = min((item.available_date + timedelta(days=next(method.confirmation_delay_days for method in config.methods if method.method_id == monitoring.method_id)) for item in confirmed_true), default=None)
    first_action = min(action_dates, default=None)
    delay = lambda value: (value - onset).days if value is not None and onset is not None else None
    return ScenarioMethodResult(draw.scenario_id, draw.scenario_seed, monitoring.method_id, monitoring.method_kind, issue.mechanism if issue else "none", "|".join(issue.zone_ids) if issue else "", sum(farm.get_zone(zone_id).area_m2 / 10_000.0 * fraction for zone_id, fraction in draw.affected_fractions.items()), len(monitoring.surveys), sum(item.completed for item in monitoring.surveys), sum(item.status == "unavailable" for item in monitoring.detections), counts["true_positives"], counts["false_positives"], counts["false_negatives"], counts["true_negatives"], bool(true_flags), delay(first_detection), delay(first_confirmation), delay(first_action), avoided_twso, monitoring.operating_cost, confirmation_cost, treatment_cost, false_positive_cost, 0.0, crop_loss_cost, total, untreated_crop_cost - total)


def _apply_legacy_compatibility(config: ExperimentConfig, rows: tuple[ScenarioMethodResult, ...], daily_truth: pd.DataFrame, farm: Farm, baseline: Mapping[str, float]) -> tuple[ScenarioMethodResult, ...]:
    if config.compatibility_phase11_config is None:
        return rows
    from src.simulation.scenarios import load_phase11_config, run_scenarios

    legacy_config = load_phase11_config(config.compatibility_phase11_config)
    legacy_config = replace(legacy_config, scenarios=replace(legacy_config.scenarios, count=config.scenario_count, seed=config.seed))
    legacy = run_scenarios(legacy_config, daily_truth, farm=farm, baseline_twso_kg_ha=baseline)
    legacy_by_scenario = {item.scenario_id: item for item in legacy.scenarios}
    updated: list[ScenarioMethodResult] = []
    for row in rows:
        source = legacy_by_scenario[row.scenario_id]
        if row.method_kind == "drone":
            avoided = source.drone_avoided_twso_kg
            net = source.drone_net_benefit
        elif row.method_kind == "ground_scout":
            avoided = source.scout_avoided_twso_kg
            net = source.scout_net_benefit
        else:
            raise ValueError("Phase 11 compatibility supports only drone and ground_scout")
        untreated_total = row.total_cost + row.net_benefit_vs_no_intervention
        target_total = untreated_total - net
        adjustment = target_total - row.total_cost
        updated.append(replace(row, avoided_twso_kg=avoided, compatibility_adjustment_cost=adjustment, total_cost=target_total, net_benefit_vs_no_intervention=net))
    return tuple(updated)


def _pairwise(rows: Sequence[ScenarioMethodResult], reference_id: str) -> tuple[PairwiseResult, ...]:
    by_scenario: dict[str, dict[str, ScenarioMethodResult]] = {}
    for row in rows:
        by_scenario.setdefault(row.scenario_id, {})[row.method_id] = row
    output: list[PairwiseResult] = []
    for scenario_id, methods in sorted(by_scenario.items()):
        reference = methods[reference_id]
        for candidate_id, candidate in sorted(methods.items()):
            if candidate_id == reference_id:
                continue
            output.append(_pair(scenario_id, candidate, reference_id, reference))
            output.append(PairwiseResult(scenario_id, candidate_id, "no_intervention", None, candidate.avoided_twso_kg, -candidate.net_benefit_vs_no_intervention, candidate.net_benefit_vs_no_intervention, candidate.net_benefit_vs_no_intervention > 0))
    return tuple(output)


def _pair(scenario_id: str, candidate: ScenarioMethodResult, reference_id: str, reference: ScenarioMethodResult) -> PairwiseResult:
    if candidate.first_detection_delay_days is None or reference.first_detection_delay_days is None:
        delay = None
    else:
        delay = float(candidate.first_detection_delay_days - reference.first_detection_delay_days)
    delta = candidate.net_benefit_vs_no_intervention - reference.net_benefit_vs_no_intervention
    return PairwiseResult(scenario_id, candidate.method_id, reference_id, delay, candidate.avoided_twso_kg - reference.avoided_twso_kg, candidate.total_cost - reference.total_cost, delta, delta > 0)


def _distributions(method_rows: Sequence[ScenarioMethodResult], pair_rows: Sequence[PairwiseResult]) -> tuple[DistributionSummary, ...]:
    records: list[DistributionSummary] = []
    method_metrics = ("first_detection_delay_days", "avoided_twso_kg", "operating_cost", "total_cost", "net_benefit_vs_no_intervention")
    for method_id in sorted({row.method_id for row in method_rows}):
        selected = [row for row in method_rows if row.method_id == method_id]
        for metric in method_metrics:
            records.append(_summary("method", method_id, metric, [getattr(row, metric) for row in selected]))
    pair_metrics = ("detection_delay_delta_days", "avoided_twso_delta_kg", "total_cost_delta", "net_benefit_delta")
    keys = sorted({(row.candidate_method_id, row.reference_method_id) for row in pair_rows})
    for candidate, reference in keys:
        selected = [row for row in pair_rows if row.candidate_method_id == candidate and row.reference_method_id == reference]
        for metric in pair_metrics:
            records.append(_summary("comparison", f"{candidate}_vs_{reference}", metric, [getattr(row, metric) for row in selected]))
    return tuple(records)


def _summary(scope: str, result_id: str, metric: str, values: Sequence[float | int | None]) -> DistributionSummary:
    numeric = sorted(float(value) for value in values if value is not None)
    if not numeric:
        return DistributionSummary(scope, result_id, metric, 0, 0, 0, 0, 0, 0)
    return DistributionSummary(scope, result_id, metric, len(numeric), mean(numeric), median(numeric), _percentile(numeric, 0.05), _percentile(numeric, 0.95), sum(value > 0 for value in numeric) / len(numeric))


def _sensitivities(method_rows: Sequence[ScenarioMethodResult], pair_rows: Sequence[PairwiseResult], parameters: Mapping[str, Mapping[str, float]]) -> tuple[SensitivityResult, ...]:
    output: list[SensitivityResult] = []
    for method_id in sorted({row.method_id for row in method_rows}):
        selected = sorted((row for row in method_rows if row.method_id == method_id), key=lambda row: row.scenario_id)
        output.extend(_sensitivity_group("method", method_id, "net_benefit_vs_no_intervention", selected, parameters, lambda row: row.net_benefit_vs_no_intervention))
    for candidate, reference in sorted({(row.candidate_method_id, row.reference_method_id) for row in pair_rows if row.reference_method_id != "no_intervention"}):
        selected = sorted((row for row in pair_rows if row.candidate_method_id == candidate and row.reference_method_id == reference), key=lambda row: row.scenario_id)
        output.extend(_sensitivity_group("comparison", f"{candidate}_vs_{reference}", "net_benefit_delta", selected, parameters, lambda row: row.net_benefit_delta))
    return tuple(output)


def _sensitivity_group(scope: str, result_id: str, target: str, rows: Sequence[object], parameters: Mapping[str, Mapping[str, float]], getter: object) -> list[SensitivityResult]:
    if not rows:
        return []
    scenario_ids = [row.scenario_id for row in rows]
    targets = [float(getter(row)) for row in rows]
    return [SensitivityResult(scope, result_id, target, name, _correlation([parameters[scenario_id][name] for scenario_id in scenario_ids], targets)) for name in next(iter(parameters.values()))]


def _scenario_parameters(draw: ScenarioDraw) -> Mapping[str, float]:
    issue = draw.issue
    return {"footprint_zone_count": float(len(issue.zone_ids) if issue else 0), "onset_day": float(issue.onset_day if issue else 0), "progression_per_day": issue.progression_per_day if issue else 0.0, "max_severity": issue.max_severity if issue else 0.0, "visibility_delay_days": float(issue.visibility_delay_days if issue else 0), "visibility_scale": issue.visibility_scale if issue else 0.0, "untreated_loss_fraction": issue.untreated_loss_fraction if issue else 0.0, "response_delay_days": float(draw.response_delay_days), "efficacy_multiplier": draw.efficacy_multiplier, "treatment_cost_multiplier": draw.treatment_cost_multiplier, "twso_value_multiplier": draw.twso_value_multiplier}


def _contiguous_zones(random: Random, farm: Farm, count: int) -> tuple[str, ...]:
    positions = {(zone.row, zone.column): zone for zone in farm.zones}
    first = random.choice(tuple(sorted(farm.zones, key=lambda zone: (zone.row, zone.column))))
    selected = {(first.row, first.column)}
    while len(selected) < count:
        candidates = {
            position
            for row, column in selected
            for position in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1))
            if position in positions and position not in selected
        }
        selected.add(random.choice(tuple(sorted(candidates))))
    return tuple(positions[position].zone_id for position in sorted(selected))


def _sample_number(random: Random, value: NumberRange) -> float:
    return random.uniform(value.minimum, value.maximum)


def _sample_integer(random: Random, value: IntegerRange) -> int:
    return random.randint(value.minimum, value.maximum)


def _percentile(values: Sequence[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _correlation(first: Sequence[float], second: Sequence[float]) -> float:
    first_mean, second_mean = mean(first), mean(second)
    numerator = sum((x - first_mean) * (y - second_mean) for x, y in zip(first, second))
    first_sum = sum((x - first_mean) ** 2 for x in first)
    second_sum = sum((y - second_mean) ** 2 for y in second)
    return 0.0 if first_sum == 0 or second_sum == 0 else max(-1.0, min(1.0, numerator / sqrt(first_sum * second_sum)))


def _keyed_seed(*parts: object) -> int:
    from hashlib import sha256
    return int.from_bytes(sha256("|".join(str(part) for part in parts).encode()).digest()[:8], "big")
