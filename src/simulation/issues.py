"""Generic crop-issue progression and intervention effects.

The functions in this module intentionally model synthetic decision value, not
crop physiology.  An issue supplies its own untreated yield-proxy loss, while
daily severity determines when that loss accrues.  An intervention can prevent
some future loss, but it never restores damage that has already accumulated.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Iterable, Mapping


ISSUE_MECHANISMS = (
    "water_deficit",
    "excess_water",
    "nutrient_deficit",
    "canopy_damage",
)

ISSUE_COLUMNS = (
    "issue_id",
    "mechanism",
    "zone_id",
    "onset_day",
    "progression_per_day",
    "max_severity",
    "visibility_delay_days",
    "visibility_scale",
    "untreated_loss_fraction",
)


class IssueConfigError(ValueError):
    """Raised when an issue CSV is invalid."""


@dataclass(frozen=True)
class IssueScenario:
    """One generic issue affecting one or more farm zones.

    All day values are zero-based campaign days. ``progression_per_day`` and
    ``max_severity`` are fractions. ``visibility_scale`` controls how visible
    the issue can become to a future observation model and does not influence
    the damage calculation.
    """

    issue_id: str
    mechanism: str
    zone_ids: tuple[str, ...]
    onset_day: int
    progression_per_day: float
    max_severity: float
    visibility_delay_days: int
    visibility_scale: float
    untreated_loss_fraction: float

    def __post_init__(self) -> None:
        if not self.issue_id.strip() or "|" in self.issue_id:
            raise ValueError("issue_id must be nonempty and must not contain '|'")
        if self.mechanism not in ISSUE_MECHANISMS:
            raise ValueError(
                f"{self.issue_id}: mechanism must be one of {ISSUE_MECHANISMS}"
            )
        if not self.zone_ids:
            raise ValueError(f"{self.issue_id}: zone_ids must not be empty")
        if len(set(self.zone_ids)) != len(self.zone_ids):
            raise ValueError(f"{self.issue_id}: zone_ids must be unique")
        if any(not zone_id.strip() for zone_id in self.zone_ids):
            raise ValueError(f"{self.issue_id}: zone_ids must not contain blanks")
        _nonnegative_integer(self.onset_day, f"{self.issue_id}: onset_day")
        _positive_fraction(
            self.progression_per_day,
            f"{self.issue_id}: progression_per_day",
        )
        _positive_fraction(self.max_severity, f"{self.issue_id}: max_severity")
        _nonnegative_integer(
            self.visibility_delay_days,
            f"{self.issue_id}: visibility_delay_days",
        )
        _positive_fraction(
            self.visibility_scale,
            f"{self.issue_id}: visibility_scale",
        )
        _positive_fraction(
            self.untreated_loss_fraction,
            f"{self.issue_id}: untreated_loss_fraction",
        )

    def severity_on(self, day: int) -> float:
        """Return issue severity for a zero-based campaign day."""

        _nonnegative_integer(day, "day")
        if day < self.onset_day:
            return 0.0
        days_active = day - self.onset_day + 1
        return min(self.max_severity, days_active * self.progression_per_day)

    def visibility_on(self, day: int) -> float:
        """Return a bounded synthetic visibility signal for a campaign day."""

        _nonnegative_integer(day, "day")
        visible_from = self.onset_day + self.visibility_delay_days
        if day < visible_from:
            return 0.0
        return self.visibility_scale * self.severity_on(day) / self.max_severity


@dataclass(frozen=True)
class InterventionRule:
    """Action limits and costs for one issue mechanism."""

    mechanism: str
    response_delay_days: int
    efficacy: float
    cutoff_day: int
    cost_per_ha: float

    def __post_init__(self) -> None:
        if self.mechanism not in ISSUE_MECHANISMS:
            raise ValueError(f"mechanism must be one of {ISSUE_MECHANISMS}")
        _nonnegative_integer(self.response_delay_days, "response_delay_days")
        _unit_fraction(self.efficacy, "efficacy")
        _nonnegative_integer(self.cutoff_day, "cutoff_day")
        if not isfinite(self.cost_per_ha) or self.cost_per_ha < 0:
            raise ValueError("cost_per_ha must be finite and nonnegative")


@dataclass(frozen=True)
class IssueOutcome:
    """Yield-proxy loss resulting from one optional intervention."""

    untreated_loss_fraction: float
    treated_loss_fraction: float
    avoided_loss_fraction: float
    accrued_loss_fraction: float
    action_day: int | None
    effective_day: int | None
    action_effective: bool
    action_cost_per_ha: float


def load_issue_scenarios(
    path: str | Path,
    *,
    valid_zone_ids: Iterable[str],
    campaign_days: int,
) -> tuple[IssueScenario, ...]:
    """Load one CSV row per issue-zone pair into grouped issue scenarios."""

    if not isinstance(campaign_days, int) or isinstance(campaign_days, bool):
        raise IssueConfigError("campaign_days must be a positive integer")
    if campaign_days < 1:
        raise IssueConfigError("campaign_days must be a positive integer")
    source_path = Path(path)
    try:
        with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise IssueConfigError(f"issue CSV {source_path} has no header")
            missing = set(ISSUE_COLUMNS) - set(reader.fieldnames)
            extra = set(reader.fieldnames) - set(ISSUE_COLUMNS)
            if missing or extra:
                raise IssueConfigError(
                    "issue CSV columns do not match schema; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
            rows = tuple(reader)
    except OSError as exc:
        raise IssueConfigError(f"unable to read issue CSV {source_path}: {exc}") from exc
    if not rows:
        raise IssueConfigError(f"issue CSV {source_path} contains no issues")

    zones = set(valid_zone_ids)
    definitions: dict[str, tuple[object, ...]] = {}
    footprints: dict[str, list[str]] = defaultdict(list)
    assigned_zones: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        context = f"issue CSV line {line_number}"
        try:
            issue_id = _row_text(row, "issue_id", context)
            mechanism = _row_text(row, "mechanism", context)
            zone_id = _row_text(row, "zone_id", context)
            definition = (
                mechanism,
                _row_int(row, "onset_day", context),
                _row_float(row, "progression_per_day", context),
                _row_float(row, "max_severity", context),
                _row_int(row, "visibility_delay_days", context),
                _row_float(row, "visibility_scale", context),
                _row_float(row, "untreated_loss_fraction", context),
            )
        except ValueError as exc:
            if isinstance(exc, IssueConfigError):
                raise
            raise IssueConfigError(f"{context}: {exc}") from exc
        if zone_id not in zones:
            raise IssueConfigError(f"{issue_id}: unknown zone_id {zone_id!r}")
        if zone_id in assigned_zones:
            raise IssueConfigError(
                f"{zone_id} has more than one issue; initial Phase 4 requires "
                "non-overlapping issue footprints"
            )
        if issue_id in definitions and definitions[issue_id] != definition:
            raise IssueConfigError(
                f"{issue_id}: footprint rows must use identical issue settings"
            )
        definitions[issue_id] = definition
        footprints[issue_id].append(zone_id)
        assigned_zones.add(zone_id)

    scenarios: list[IssueScenario] = []
    for issue_id, definition in definitions.items():
        mechanism, onset, progression, maximum, delay, scale, loss = definition
        try:
            scenario = IssueScenario(
                issue_id=issue_id,
                mechanism=str(mechanism),
                zone_ids=tuple(footprints[issue_id]),
                onset_day=int(onset),
                progression_per_day=float(progression),
                max_severity=float(maximum),
                visibility_delay_days=int(delay),
                visibility_scale=float(scale),
                untreated_loss_fraction=float(loss),
            )
        except ValueError as exc:
            raise IssueConfigError(str(exc)) from exc
        if scenario.onset_day >= campaign_days:
            raise IssueConfigError(
                f"{scenario.issue_id}: onset_day must fall within the campaign"
            )
        scenarios.append(scenario)
    return tuple(scenarios)


def evaluate_intervention(
    issue: IssueScenario,
    *,
    campaign_days: int,
    action_day: int | None = None,
    rule: InterventionRule | None = None,
) -> IssueOutcome:
    """Calculate loss with no action or one action against an issue.

    Severity is used as a relative daily damage weight. The weights are scaled
    so a completely untreated campaign reaches ``untreated_loss_fraction``.
    Effective treatment reduces only damage from ``effective_day`` onward.
    """

    if not isinstance(campaign_days, int) or isinstance(campaign_days, bool):
        raise ValueError("campaign_days must be a positive integer")
    if campaign_days < 1:
        raise ValueError("campaign_days must be a positive integer")
    if issue.onset_day >= campaign_days:
        raise ValueError("issue onset_day must fall within the campaign")

    if action_day is None:
        if rule is not None:
            raise ValueError("rule requires an action_day")
        effective_day = None
        action_effective = False
        action_cost = 0.0
    else:
        _nonnegative_integer(action_day, "action_day")
        if rule is None:
            raise ValueError("action_day requires an intervention rule")
        if rule.mechanism != issue.mechanism:
            raise ValueError("intervention mechanism does not match the issue")
        effective_day = action_day + rule.response_delay_days
        action_effective = (
            action_day < campaign_days
            and effective_day < campaign_days
            and effective_day <= rule.cutoff_day
        )
        action_cost = rule.cost_per_ha

    weights = [issue.severity_on(day) for day in range(campaign_days)]
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("issue must accrue positive damage during the campaign")

    if action_effective:
        assert effective_day is not None
        accrued_weight = sum(weights[:effective_day])
        future_weight = total_weight - accrued_weight
        treated_weight = accrued_weight + future_weight * (1.0 - rule.efficacy)
    else:
        accrued_weight = total_weight
        treated_weight = total_weight

    untreated_loss = issue.untreated_loss_fraction
    accrued_loss = untreated_loss * accrued_weight / total_weight
    treated_loss = untreated_loss * treated_weight / total_weight
    avoided_loss = untreated_loss - treated_loss
    return IssueOutcome(
        untreated_loss_fraction=untreated_loss,
        treated_loss_fraction=treated_loss,
        avoided_loss_fraction=avoided_loss,
        accrued_loss_fraction=accrued_loss,
        action_day=action_day,
        effective_day=effective_day,
        action_effective=action_effective,
        action_cost_per_ha=action_cost,
    )


def _nonnegative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _positive_fraction(value: float, name: str) -> None:
    if not isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be greater than 0 and at most 1")


def _unit_fraction(value: float, name: str) -> None:
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _row_text(row: Mapping[str, str], key: str, context: str) -> str:
    value = row.get(key)
    if value is None or not value.strip():
        raise IssueConfigError(f"{context}: {key} must not be empty")
    return value.strip()


def _row_int(row: Mapping[str, str], key: str, context: str) -> int:
    value = _row_text(row, key, context)
    try:
        return int(value)
    except ValueError as exc:
        raise IssueConfigError(f"{context}: {key} must be an integer") from exc


def _row_float(row: Mapping[str, str], key: str, context: str) -> float:
    value = _row_text(row, key, context)
    try:
        return float(value)
    except ValueError as exc:
        raise IssueConfigError(f"{context}: {key} must be numeric") from exc
