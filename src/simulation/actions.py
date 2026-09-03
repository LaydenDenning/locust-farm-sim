"""Treatment actions and paired biological outcomes for Phase 9."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.simulation.comparison import METHODS
from src.simulation.confirmation import (
    ConfirmationEvaluation,
    Phase8Config,
    Phase8Result,
    load_phase8_config,
)
from src.simulation.issues import (
    ISSUE_MECHANISMS,
    InterventionRule,
    IssueScenario,
    evaluate_intervention,
)


STRATEGIES = ("no_intervention", "scout", "drone")
ACTION_EVALUATIONS = ("appropriate_action", "unnecessary_action")


class ActionConfigError(ValueError):
    """Raised when Phase 9 configuration is invalid."""


@dataclass(frozen=True)
class Phase9OutputConfig:
    directory: Path
    action_events_filename: str
    action_evaluation_filename: str
    scenario_outcomes_filename: str

    def __post_init__(self) -> None:
        for name in (
            "action_events_filename",
            "action_evaluation_filename",
            "scenario_outcomes_filename",
        ):
            value = getattr(self, name)
            if not value.strip() or Path(value).name != value:
                raise ValueError(f"output.{name} must be a simple file name")


@dataclass(frozen=True)
class Phase9Config:
    source_path: Path
    phase8: Phase8Config
    intervention_rules: tuple[InterventionRule, ...]
    output: Phase9OutputConfig

    def rule_for(self, mechanism: str) -> InterventionRule:
        for rule in self.intervention_rules:
            if rule.mechanism == mechanism:
                return rule
        raise ValueError(f"no intervention rule for {mechanism}")


@dataclass(frozen=True)
class ActionEvent:
    """Truth-blind action triggered by one confirmed finding."""

    action_id: str
    method: str
    confirmation_id: str
    survey_id: str
    zone_id: str
    confirmation_date: date
    action_date: date
    action: str

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.action != "treat":
            raise ValueError("action must be treat")
        if self.action_date < self.confirmation_date:
            raise ValueError("action_date must not precede confirmation_date")


@dataclass(frozen=True)
class ActionEvaluation:
    """Truth-bearing evaluation kept separate from operational actions."""

    action_id: str
    truth_positive: bool
    issue_id: str
    mechanism: str
    evaluation: str

    def __post_init__(self) -> None:
        if self.evaluation not in ACTION_EVALUATIONS:
            raise ValueError(f"evaluation must be one of {ACTION_EVALUATIONS}")


@dataclass(frozen=True)
class ScenarioOutcome:
    """One issue-zone outcome under one scouting strategy."""

    issue_id: str
    mechanism: str
    zone_id: str
    strategy: str
    onset_date: date
    confirmation_date: date | None
    action_date: date | None
    effective_date: date | None
    action_effective: bool
    untreated_loss_fraction: float
    treated_loss_fraction: float
    avoided_loss_fraction: float
    accrued_loss_fraction: float
    action_cost_per_ha: float

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if self.action_date is not None and self.confirmation_date is None:
            raise ValueError("an action requires a confirmation date")
        if (
            self.action_date is not None
            and self.confirmation_date is not None
            and self.action_date < self.confirmation_date
        ):
            raise ValueError("action_date must not precede confirmation_date")


@dataclass(frozen=True)
class Phase9Result:
    action_events: tuple[ActionEvent, ...]
    action_evaluations: tuple[ActionEvaluation, ...]
    scenario_outcomes: tuple[ScenarioOutcome, ...]


def load_phase9_config(path: str | Path) -> Phase9Config:
    """Load strict Phase 9 action and intervention settings."""

    source_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ActionConfigError(f"unable to read {source_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ActionConfigError(f"invalid YAML in {source_path}: {exc}") from exc

    root = _mapping(raw, "configuration")
    _exact_keys(root, {"phase8_config", "interventions", "output"}, "configuration")
    phase8 = load_phase8_config(
        _resolve_path(source_path.parent, root["phase8_config"])
    )
    interventions = _mapping(root["interventions"], "interventions")
    _exact_keys(interventions, set(ISSUE_MECHANISMS), "interventions")

    rules: list[InterventionRule] = []
    try:
        for mechanism in ISSUE_MECHANISMS:
            context = f"interventions.{mechanism}"
            values = _mapping(interventions[mechanism], context)
            _exact_keys(
                values,
                {"response_delay_days", "efficacy", "cutoff_day", "cost_per_ha"},
                context,
            )
            rules.append(
                InterventionRule(
                    mechanism=mechanism,
                    response_delay_days=_integer(
                        values["response_delay_days"], f"{context}.response_delay_days"
                    ),
                    efficacy=_number(values["efficacy"], f"{context}.efficacy"),
                    cutoff_day=_integer(values["cutoff_day"], f"{context}.cutoff_day"),
                    cost_per_ha=_number(
                        values["cost_per_ha"], f"{context}.cost_per_ha"
                    ),
                )
            )

        output = _mapping(root["output"], "output")
        _exact_keys(
            output,
            {
                "directory",
                "action_events_filename",
                "action_evaluation_filename",
                "scenario_outcomes_filename",
            },
            "output",
        )
        output_config = Phase9OutputConfig(
            directory=_resolve_path(source_path.parent, output["directory"]),
            action_events_filename=_text(
                output["action_events_filename"], "output.action_events_filename"
            ),
            action_evaluation_filename=_text(
                output["action_evaluation_filename"],
                "output.action_evaluation_filename",
            ),
            scenario_outcomes_filename=_text(
                output["scenario_outcomes_filename"],
                "output.scenario_outcomes_filename",
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ActionConfigError):
            raise
        raise ActionConfigError(str(exc)) from exc
    return Phase9Config(source_path, phase8, tuple(rules), output_config)


def simulate_actions(config: Phase9Config, phase8_result: Phase8Result) -> Phase9Result:
    """Turn confirmed findings into actions, then evaluate paired outcomes."""

    confirmation_evaluations = _confirmation_evaluations(phase8_result.evaluations)
    issues = config.phase8.phase7.phase5.phase4.issues
    issue_by_zone = {
        zone_id: issue for issue in issues for zone_id in issue.zone_ids
    }
    events: list[ActionEvent] = []
    evaluations: list[ActionEvaluation] = []
    confirmed = sorted(
        (item for item in phase8_result.confirmation_events if item.result == "confirmed"),
        key=lambda item: (
            item.confirmation_date,
            item.method,
            item.confirmation_id,
        ),
    )
    first_confirmed = []
    seen_method_zones: set[tuple[str, str]] = set()
    for item in confirmed:
        key = (item.method, item.zone_id)
        if key not in seen_method_zones:
            first_confirmed.append(item)
            seen_method_zones.add(key)

    for index, item in enumerate(first_confirmed, start=1):
        evaluation = confirmation_evaluations.get(item.confirmation_id)
        if evaluation is None:
            raise ValueError(
                f"confirmation has no Phase 8 evaluation: {item.confirmation_id}"
            )
        action_id = f"ACTION_{index:04d}"
        events.append(
            ActionEvent(
                action_id=action_id,
                method=item.method,
                confirmation_id=item.confirmation_id,
                survey_id=item.survey_id,
                zone_id=item.zone_id,
                confirmation_date=item.confirmation_date,
                action_date=item.confirmation_date,
                action="treat",
            )
        )
        issue = issue_by_zone.get(item.zone_id) if evaluation.truth_positive else None
        evaluations.append(
            ActionEvaluation(
                action_id=action_id,
                truth_positive=evaluation.truth_positive,
                issue_id=issue.issue_id if issue else "",
                mechanism=issue.mechanism if issue else "",
                evaluation=(
                    "appropriate_action"
                    if evaluation.truth_positive
                    else "unnecessary_action"
                ),
            )
        )

    outcomes = _scenario_outcomes(config, issues, events, evaluations)
    return Phase9Result(tuple(events), tuple(evaluations), outcomes)


def _scenario_outcomes(
    config: Phase9Config,
    issues: Sequence[IssueScenario],
    actions: Sequence[ActionEvent],
    evaluations: Sequence[ActionEvaluation],
) -> tuple[ScenarioOutcome, ...]:
    evaluation_by_action = {item.action_id: item for item in evaluations}
    sowing_date = config.phase8.phase7.phase5.phase4.phase1.calendar.base_sowing_date
    campaign_days = (
        config.phase8.phase7.phase5.phase4.phase1.calendar.max_duration_days
    )
    outcomes: list[ScenarioOutcome] = []
    for issue in issues:
        onset_date = sowing_date + timedelta(days=issue.onset_day)
        for zone_id in issue.zone_ids:
            for strategy in STRATEGIES:
                action = _first_appropriate_action(
                    actions,
                    evaluation_by_action,
                    issue=issue,
                    zone_id=zone_id,
                    method=None if strategy == "no_intervention" else strategy,
                    onset_date=onset_date,
                )
                if action is None:
                    result = evaluate_intervention(issue, campaign_days=campaign_days)
                    effective_date = None
                else:
                    action_day = (action.action_date - sowing_date).days
                    rule = config.rule_for(issue.mechanism)
                    result = evaluate_intervention(
                        issue,
                        campaign_days=campaign_days,
                        action_day=action_day,
                        rule=rule,
                    )
                    effective_date = (
                        sowing_date + timedelta(days=result.effective_day)
                        if result.effective_day is not None
                        else None
                    )
                outcomes.append(
                    ScenarioOutcome(
                        issue_id=issue.issue_id,
                        mechanism=issue.mechanism,
                        zone_id=zone_id,
                        strategy=strategy,
                        onset_date=onset_date,
                        confirmation_date=(action.confirmation_date if action else None),
                        action_date=action.action_date if action else None,
                        effective_date=effective_date,
                        action_effective=result.action_effective,
                        untreated_loss_fraction=result.untreated_loss_fraction,
                        treated_loss_fraction=result.treated_loss_fraction,
                        avoided_loss_fraction=result.avoided_loss_fraction,
                        accrued_loss_fraction=result.accrued_loss_fraction,
                        action_cost_per_ha=result.action_cost_per_ha,
                    )
                )
    return tuple(outcomes)


def _first_appropriate_action(
    actions: Sequence[ActionEvent],
    evaluations: Mapping[str, ActionEvaluation],
    *,
    issue: IssueScenario,
    zone_id: str,
    method: str | None,
    onset_date: date,
) -> ActionEvent | None:
    if method is None:
        return None
    eligible = [
        item
        for item in actions
        if item.method == method
        and item.zone_id == zone_id
        and item.action_date >= onset_date
        and evaluations[item.action_id].truth_positive
        and evaluations[item.action_id].issue_id == issue.issue_id
    ]
    return min(
        eligible,
        key=lambda item: (item.action_date, item.action_id),
        default=None,
    )


def _confirmation_evaluations(
    evaluations: Sequence[ConfirmationEvaluation],
) -> dict[str, ConfirmationEvaluation]:
    indexed = {item.confirmation_id: item for item in evaluations}
    if len(indexed) != len(evaluations):
        raise ValueError("Phase 8 contains duplicate confirmation evaluation IDs")
    return indexed


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActionConfigError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise ActionConfigError(
            f"{name} keys do not match schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _resolve_path(parent: Path, value: Any) -> Path:
    text = _text(value, "path")
    path = Path(text)
    return (parent / path).resolve() if not path.is_absolute() else path.resolve()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ActionConfigError(f"{name} must be nonempty text")
    return value.strip()


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ActionConfigError(f"{name} must be an integer")
    return value


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ActionConfigError(f"{name} must be numeric")
    return float(value)
