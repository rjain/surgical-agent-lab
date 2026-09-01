"""Smoke tests for the supplied Lab 1 modules.

These guard the three properties of the raw data that are easy to get wrong,
plus the shape of what the rules engine returns. Run with::

    pytest -q
"""

from __future__ import annotations

import pytest

from lab.data import (
    Case,
    list_cases,
    load_case,
    overlapping_tools,
    to_seconds,
    tool_changes_within,
)
from lab.metrics import cohort_medians, get_metrics, session_summary, step_metrics
from lab.rules import RULES, find_deviations, list_deviations

GOLDEN = "case_045"
MULTIPART = "case_036"


@pytest.fixture(scope="module")
def golden() -> Case:
    return load_case(GOLDEN)


# --- time parsing -----------------------------------------------------------


def test_to_seconds_parses_both_formats():
    assert to_seconds("01:45:45.519000") == pytest.approx(6345.519)
    assert to_seconds("00:00:01") == pytest.approx(1.0)
    assert to_seconds(195.689385) == pytest.approx(195.689385)


def test_to_seconds_returns_none_rather_than_raising():
    for bad in (None, "", "not a time", float("nan")):
        assert to_seconds(bad) is None


# --- the three data traps ---------------------------------------------------


def test_durations_are_never_negative_even_on_multipart_cases():
    """The bug this whole module exists to prevent.

    Times restart at zero in each video part, so naive subtraction across a
    part boundary yields a negative duration that looks plausible.
    """
    case = load_case(MULTIPART)
    assert case.is_multipart, f"{MULTIPART} was expected to span several parts"
    assert (case.tasks["duration_s"] > 0).all()
    assert (case.tools["duration_s"] > 0).all()
    assert session_summary(case)["labelled_s"] > 0


def test_duplicate_tool_rows_are_dropped(golden):
    import pandas as pd

    from lab.data import DATA_DIR

    raw = pd.read_csv(DATA_DIR / GOLDEN / "tools.csv")
    assert len(golden.tools) < len(raw), "expected duplicates to be removed"


def test_unknown_instruments_are_kept_not_dropped():
    """A blank tool name means something was mounted and not identified.

    That is a finding, not a row to discard.
    """
    total = sum(
        int((load_case(c).tools["tool"] == "<unknown>").sum())
        for c in (GOLDEN, MULTIPART)
    )
    assert total > 0


def test_overlap_helpers_respect_part_boundaries():
    case = load_case(MULTIPART)
    first, second = case.parts[0], case.parts[1]
    window = case.tasks[case.tasks["part"] == first].iloc[0]
    hits = overlapping_tools(case.tools, first, window.start_s, window.end_s)
    assert set(hits["part"].unique()) <= {first}
    assert second not in set(hits["part"].unique())


def test_tool_changes_is_stricter_than_overlap(golden):
    seg = golden.tasks.iloc[0]
    changes = tool_changes_within(golden.tools, seg.part, seg.start_s, seg.end_s)
    during = overlapping_tools(golden.tools, seg.part, seg.start_s, seg.end_s)
    assert len(changes) <= len(during)


# --- metrics ----------------------------------------------------------------


def test_cohort_medians_cover_the_named_steps():
    medians = cohort_medians()
    assert "Suturing" in medians
    assert all(v > 0 for v in medians.values())


def test_step_metrics_has_a_row_per_segment(golden):
    metrics = step_metrics(golden)
    assert len(metrics) == len(golden.tasks)
    assert (metrics["duration_s"] > 0).all()


def test_get_metrics_reports_a_useful_error_for_an_unknown_step():
    result = get_metrics(GOLDEN, step="Not A Real Step")
    assert "error" in result
    assert result["available_steps"], "the error should say what is available"


# --- rules ------------------------------------------------------------------


def test_golden_case_produces_flags():
    """If this fails the lab has no payoff: there is nothing to explain."""
    found = find_deviations(GOLDEN)
    assert len(found) >= 3, f"{GOLDEN} produced only {len(found)} flags"


def test_every_rule_fires_somewhere_in_the_sample_set():
    """A rule that never fires is dead weight in the room."""
    fired = {
        d.rule_id
        for case in (GOLDEN, MULTIPART, "case_008", "case_129")
        for d in find_deviations(case)
    }
    expected = {r.__name__ for r in RULES}
    missing = {
        "swap_rate_outliers": "swap_rate",
        "step_overruns": "step_overrun",
        "step_oscillations": "step_oscillation",
        "unknown_instruments": "unknown_instrument",
    }
    for func_name in expected:
        assert missing[func_name] in fired, f"{func_name} never fired"


def test_every_flag_carries_traceable_evidence():
    for dev in find_deviations(GOLDEN):
        assert dev.evidence.strip(), "a flag without evidence cannot be defended"
        assert any(ch.isdigit() for ch in dev.evidence) or "not identified" in dev.evidence
        assert 0.0 <= dev.score <= 1.0


def test_flags_use_efficiency_vocabulary_not_clinical():
    """The field is a score. It must not acquire clinical connotations."""
    for dev in find_deviations(GOLDEN):
        keys = dev.as_dict().keys()
        assert "severity" not in keys and "risk" not in keys


def test_list_deviations_is_json_safe():
    import json

    json.dumps(list_deviations(GOLDEN))


def test_the_dataset_is_actually_present():
    cases = list_cases()
    assert len(cases) > 0, "no cases found — is LAB_DATA_DIR set correctly?"
