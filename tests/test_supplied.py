"""Smoke tests for the supplied modules — Lab 1, plus the shared plumbing.

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


# --- schemas the Gemini API will actually accept ----------------------------


def test_schemas_avoid_types_the_api_rejects():
    """Tuples serialise to `prefixItems`, which the Gemini API refuses.

    The request fails while being built, so this is caught before any network
    call — but only if something checks. That something is this test.
    """
    from lab.lab2_analyze import TechniqueNotes

    def offending_keys(node, path="") -> list[str]:
        """Find `prefixItems` used as a schema key, not merely mentioned in prose.

        Docstrings end up in the schema's `description`, so a plain substring
        search over the JSON gives a false positive on this very test.
        """
        found = []
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "prefixItems":
                    found.append(path or "<root>")
                found += offending_keys(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                found += offending_keys(value, f"{path}[{i}]")
        return found

    schemas = {"TechniqueNotes": TechniqueNotes.model_json_schema()}
    try:
        from lab.variants.auditor import SessionReport

        schemas["SessionReport"] = SessionReport.model_json_schema()
    except Exception:
        pass

    for name, schema in schemas.items():
        bad = offending_keys(schema)
        assert not bad, (
            f"{name} has tuple field(s) at {bad} — the Gemini API rejects "
            "prefixItems. Use separate fields or a list."
        )


def test_technique_notes_requires_an_honesty_field():
    from lab.lab2_analyze import TechniqueNotes

    assert "not_visible" in TechniqueNotes.model_fields


# --- clip resolution --------------------------------------------------------


def test_manifest_loads_even_when_empty_or_absent():
    """The rest of the lab must still run before any clip is uploaded."""
    from lab.clips import load_manifest

    assert isinstance(load_manifest(), dict)


def test_unknown_clip_id_says_what_is_known():
    from lab.clips import ClipUnavailable, resolve_clip

    with pytest.raises(ClipUnavailable) as err:
        resolve_clip("definitely-not-a-clip")
    assert "not in the manifest" in str(err.value)


def test_unreachable_clip_explains_the_own_key_case():
    """The message has to name the cause, not just fail."""
    from lab import clips

    clips._resolved.clear()
    original = clips.load_manifest
    fake = clips.Clip(
        clip_id="x", case_id="case_045", part=1, start_s=0.0, end_s=10.0,
        uri="https://generativelanguage.googleapis.com/v1beta/files/notreal000",
        local="data/clips/nothing-here.mp4",
    )
    clips.load_manifest = lambda: {"x": fake}
    try:
        with pytest.raises(clips.ClipUnavailable) as err:
            clips.resolve_clip("x")
        message = str(err.value)
        assert "own API key" in message
        assert "data/clips" in message
    finally:
        clips.load_manifest = original
        clips._resolved.clear()


def test_every_flag_nominates_a_watchable_window():
    """A flag spanning 45 minutes is not something you can send to a model.

    Sending a whole flagged segment costs roughly 170,000 tokens. The watch
    window has to stay small and stay inside the segment it came from.
    """
    from lab.rules import FOCUS_WINDOW_S, find_deviations

    for case in ("case_045", "case_036", "case_044"):
        for dev in find_deviations(case):
            lo, hi = dev.watch_window
            assert hi > lo, f"{dev.rule_id} produced an empty window"
            assert hi - lo <= FOCUS_WINDOW_S + 0.01, (
                f"{dev.rule_id} window is {hi - lo:.0f}s, over the "
                f"{FOCUS_WINDOW_S:.0f}s budget"
            )
            assert dev.start_s <= lo and hi <= dev.end_s + 0.01, (
                f"{dev.rule_id} watch window falls outside its own segment"
            )


# --- the skeletons must stay skeletons --------------------------------------


def test_participant_files_still_need_writing():
    """Guards against a reference solution being copied in and committed.

    Testing a solution locally means overwriting these files, and forgetting to
    restore one would hand every participant the answers. Covers all six entry
    points across the four participant files — the variants included, since
    those are the ones most likely to be staged for a demo and left behind.
    """
    import lab.lab2_analyze as lab2
    import lab.lab3_agent as lab3
    from lab.variants import auditor, coach

    with pytest.raises(NotImplementedError):
        lab2.analyze_clip("case_045")
    with pytest.raises(NotImplementedError):
        lab2.validate(None, 0.0, 1.0)
    with pytest.raises(NotImplementedError):
        lab3.build_agent()
    with pytest.raises(NotImplementedError):
        coach.build_workflow_tracker()
    with pytest.raises(NotImplementedError):
        coach.build_coach()
    with pytest.raises(NotImplementedError):
        auditor.build_auditor()


def test_the_dataset_is_actually_present():
    cases = list_cases()
    assert len(cases) > 0, "no cases found — is LAB_DATA_DIR set correctly?"
