"""Unit tests for generic issue progression and intervention effects."""

from __future__ import annotations

import unittest

from src.simulation.issues import (
    ISSUE_MECHANISMS,
    InterventionRule,
    IssueScenario,
    evaluate_intervention,
)


def _issue(**overrides: object) -> IssueScenario:
    values: dict[str, object] = {
        "issue_id": "ISSUE_01",
        "mechanism": "canopy_damage",
        "zone_ids": ("Z_R2_C2", "Z_R2_C3"),
        "onset_day": 10,
        "progression_per_day": 0.05,
        "max_severity": 0.6,
        "visibility_delay_days": 3,
        "visibility_scale": 0.8,
        "untreated_loss_fraction": 0.3,
    }
    values.update(overrides)
    return IssueScenario(**values)  # type: ignore[arg-type]


def _rule(**overrides: object) -> InterventionRule:
    values: dict[str, object] = {
        "mechanism": "canopy_damage",
        "response_delay_days": 2,
        "efficacy": 0.75,
        "cutoff_day": 80,
        "cost_per_ha": 45.0,
    }
    values.update(overrides)
    return InterventionRule(**values)  # type: ignore[arg-type]


class IssueScenarioTests(unittest.TestCase):
    def test_all_planned_mechanisms_are_available(self) -> None:
        self.assertEqual(
            ISSUE_MECHANISMS,
            (
                "water_deficit",
                "excess_water",
                "nutrient_deficit",
                "canopy_damage",
            ),
        )

    def test_severity_progresses_and_stops_at_maximum(self) -> None:
        issue = _issue()
        self.assertEqual(issue.severity_on(9), 0.0)
        self.assertAlmostEqual(issue.severity_on(10), 0.05)
        self.assertAlmostEqual(issue.severity_on(14), 0.25)
        self.assertAlmostEqual(issue.severity_on(30), 0.6)

    def test_visibility_has_an_independent_delay_and_scale(self) -> None:
        issue = _issue()
        self.assertEqual(issue.visibility_on(12), 0.0)
        self.assertAlmostEqual(issue.visibility_on(13), 0.8 * 0.2 / 0.6)
        self.assertAlmostEqual(issue.visibility_on(30), 0.8)

    def test_invalid_issue_values_are_rejected(self) -> None:
        invalid = (
            {"mechanism": "heat"},
            {"zone_ids": ()},
            {"zone_ids": ("Z_R1_C1", "Z_R1_C1")},
            {"onset_day": -1},
            {"progression_per_day": 0.0},
            {"max_severity": 1.1},
            {"visibility_delay_days": -1},
            {"untreated_loss_fraction": 0.0},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                _issue(**values)


class InterventionTests(unittest.TestCase):
    def test_no_action_returns_full_untreated_loss(self) -> None:
        outcome = evaluate_intervention(_issue(), campaign_days=100)
        self.assertAlmostEqual(outcome.treated_loss_fraction, 0.3)
        self.assertAlmostEqual(outcome.avoided_loss_fraction, 0.0)
        self.assertFalse(outcome.action_effective)
        self.assertEqual(outcome.action_cost_per_ha, 0.0)

    def test_treatment_keeps_accrued_damage_and_reduces_future_damage(self) -> None:
        outcome = evaluate_intervention(
            _issue(), campaign_days=100, action_day=20, rule=_rule()
        )
        self.assertTrue(outcome.action_effective)
        self.assertEqual(outcome.effective_day, 22)
        self.assertGreater(outcome.accrued_loss_fraction, 0.0)
        self.assertGreater(outcome.treated_loss_fraction, outcome.accrued_loss_fraction)
        self.assertLess(outcome.treated_loss_fraction, outcome.untreated_loss_fraction)
        self.assertAlmostEqual(outcome.action_cost_per_ha, 45.0)

    def test_earlier_equally_effective_treatment_never_increases_loss(self) -> None:
        issue = _issue()
        rule = _rule()
        early = evaluate_intervention(
            issue, campaign_days=100, action_day=15, rule=rule
        )
        late = evaluate_intervention(
            issue, campaign_days=100, action_day=35, rule=rule
        )
        self.assertLessEqual(early.treated_loss_fraction, late.treated_loss_fraction)

    def test_action_after_cutoff_costs_money_but_prevents_no_damage(self) -> None:
        outcome = evaluate_intervention(
            _issue(),
            campaign_days=100,
            action_day=31,
            rule=_rule(response_delay_days=2, cutoff_day=32),
        )
        self.assertFalse(outcome.action_effective)
        self.assertAlmostEqual(outcome.treated_loss_fraction, 0.3)
        self.assertAlmostEqual(outcome.avoided_loss_fraction, 0.0)
        self.assertAlmostEqual(outcome.action_cost_per_ha, 45.0)

    def test_full_efficacy_before_onset_prevents_all_loss(self) -> None:
        outcome = evaluate_intervention(
            _issue(),
            campaign_days=100,
            action_day=5,
            rule=_rule(response_delay_days=0, efficacy=1.0),
        )
        self.assertAlmostEqual(outcome.accrued_loss_fraction, 0.0)
        self.assertAlmostEqual(outcome.treated_loss_fraction, 0.0)
        self.assertAlmostEqual(outcome.avoided_loss_fraction, 0.3)

    def test_mismatched_action_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            evaluate_intervention(
                _issue(),
                campaign_days=100,
                action_day=20,
                rule=_rule(mechanism="water_deficit"),
            )

    def test_action_and_rule_must_be_supplied_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires"):
            evaluate_intervention(_issue(), campaign_days=100, action_day=20)
        with self.assertRaisesRegex(ValueError, "requires"):
            evaluate_intervention(_issue(), campaign_days=100, rule=_rule())


if __name__ == "__main__":
    unittest.main()
