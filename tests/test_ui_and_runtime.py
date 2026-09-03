"""Tests for the UI and the agent runner — no key, no tokens, no network.

The variant tabs are the part of the repo a participant sees last and we test
least, so these render every one of their states against a fake agent:
unwritten, half-written, and finished. A group that finishes Variant A at
minute 140 should not be the first person to find out that the tab throws.

The fake agent is the point. It returns a scripted tool call, which lets the
trace rendering be tested for real without spending a token or needing a key.

Run with::

    pytest -q
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = str(REPO_ROOT / "ui" / "app.py")

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest


# --------------------------------------------------------------------------
# lab/runtime.py — the trace is what makes an answer checkable
# --------------------------------------------------------------------------


def test_a_reply_with_no_calls_is_not_grounded():
    from lab.runtime import Reply, ToolCall

    assert Reply(text="the step took 12 minutes").grounded is False
    assert Reply(calls=[ToolCall("get_metrics", {})]).grounded is True


def test_tool_calls_keep_their_arguments_and_response():
    from lab.runtime import ToolCall

    call = ToolCall("get_metrics", {"case_id": "case_045"})
    call.response = {"duration_s": 720.0}
    assert call.args["case_id"] == "case_045"
    assert call.response["duration_s"] == 720.0


def test_conversation_collects_the_trace_from_events(monkeypatch):
    """Drive the event loop that turns ADK events into a Reply.

    Guards the part most likely to break on an ADK upgrade: function calls and
    their responses arrive in separate events, and pairing them up is ours.
    """
    from lab import runtime

    class FakePart:
        def __init__(self, text=None, call=None, response=None):
            self.text = text
            self.function_call = call
            self.function_response = response

    class Named:
        def __init__(self, name, **kw):
            self.name = name
            self.__dict__.update(kw)

    class FakeEvent:
        def __init__(self, parts, final=False):
            self.content = Named("content", parts=parts)
            self._final = final

        def is_final_response(self):
            return self._final

    events = [
        FakeEvent([FakePart(call=Named("get_metrics", args={"case_id": "case_045"}))]),
        FakeEvent([FakePart(response=Named("get_metrics", response={"duration_s": 720.0}))]),
        FakeEvent([FakePart(text="Suturing ran 12.0 minutes.")], final=True),
    ]

    talk = object.__new__(runtime.Conversation)
    talk._session = Named("session", id="s1")
    talk._runner = Named("runner", run=lambda **kw: iter(events))

    reply = talk.ask("which step ran longest?")
    assert reply.text == "Suturing ran 12.0 minutes."
    assert [c.name for c in reply.calls] == ["get_metrics"]
    assert reply.calls[0].response == {"duration_s": 720.0}
    assert reply.grounded


# --------------------------------------------------------------------------
# ui/app.py — the tabs, in every state a participant will see
# --------------------------------------------------------------------------


@pytest.fixture
def app():
    """A fresh app run. Fails loudly if the script raised."""

    def run(**session_state):
        at = AppTest.from_file(APP, default_timeout=60)
        for key, value in session_state.items():
            at.session_state[key] = value
        at.run()
        assert not at.exception, at.exception
        return at

    return run


def test_the_app_loads_and_shows_the_three_tabs(app):
    at = app()
    assert at.title[0].value == "Surgical Workflow Deviation Auditor"
    assert len(at.tabs) == 3


def test_unwritten_variants_say_so_instead_of_crashing(app):
    """The state the repo ships in, and the state it spends most of the lab in."""
    at = app()
    said = " ".join(m.value for m in at.info)
    assert "Variant A is not wired up yet" in said
    assert "Variant B is not wired up yet" in said


def test_the_session_tab_never_needs_a_model(app):
    """Lab 1 must stand up with no key at all. It is rule-based by design."""
    at = app()
    labels = [m.label for m in at.metric]
    assert "Flagged moments" in labels
    assert not at.error


def test_a_flagged_moment_offers_the_watch_window_not_the_whole_segment(app):
    """The 16x token saving lives here; a regression would be silent and costly."""
    at = app()
    captions = " ".join(c.value for c in at.caption)
    assert "Watch window" in captions
    assert "40 seconds around the instant" in captions


# --------------------------------------------------------------------------
# the finished variants, against a fake agent
# --------------------------------------------------------------------------


def test_a_finished_coach_renders_its_answer_and_its_trace(app, monkeypatch):
    from lab.runtime import Reply, ToolCall

    call = ToolCall("get_metrics", {"case_id": "case_045", "step": "Suturing"})
    call.response = {"duration_s": 720.0, "ratio_to_median": 1.43}
    log = [
        {"role": "user", "text": "which step ran longest?"},
        {
            "role": "assistant",
            "text": "Suturing, at 12.0 minutes — 1.43x the median.",
            "calls": [call],
        },
    ]
    at = app(**{"log::case_045": log, "coach::case_045": (object(), "")})
    body = " ".join(m.value for m in at.markdown)
    assert "1.43x the median" in body
    assert any("Grounding — 1 tool call" in e.label for e in at.expander)


def test_an_ungrounded_answer_is_called_out_as_invented(app):
    """A number with no tool call behind it is the failure mode to surface."""
    log = [{"role": "assistant", "text": "It took about 12 minutes.", "calls": []}]
    at = app(**{"log::case_045": log, "coach::case_045": (object(), "")})
    captions = " ".join(c.value for c in at.caption)
    assert "No tools called" in captions
    assert "invented" in captions


def test_a_coach_without_a_tracker_still_loads_and_says_which_form(app):
    """Minimum shippable Variant A. Degrading must not look like failing."""
    at = app(**{"coach::case_045": (object(), "single-agent form")})
    captions = " ".join(c.value for c in at.caption)
    assert "single-agent form" in captions


REPORT = {
    "case_id": "case_045",
    "headline": "Session review",
    "duration_vs_cohort": "1.43x the median",
    "findings": [
        {
            "rank": 1,
            "step": "Suturing",
            "t": 8596.0,
            "rule_id": "step_overrun",
            "headline": "Prolonged execution",
            "detail": "Ran long against the median.",
            "evidence": "23.7 min against 16.5 min",
        }
    ],
    "recommendations": ["Complete the suturing sequence before transitioning."],
    "limitations": ["Foot pedal activation was not visible."],
}


@pytest.fixture
def finished_auditor(monkeypatch):
    """Stand a finished Variant B in place of the skeleton.

    Patches the module attribute rather than the file, so the app's own
    ``from lab.variants.auditor import build_auditor`` picks it up and the
    whole tab — guard, button, spinner, rendering — runs for real without a
    model.
    """
    from lab.variants import auditor

    monkeypatch.setattr(auditor, "build_auditor", lambda: (lambda case_id: REPORT))
    return REPORT


def review_button(at):
    """The Auditor tab's button, not one of the Session tab's."""
    for button in at.button:
        if button.label.startswith("Review "):
            return button
    raise AssertionError(
        "no Review button; labels were " + repr([b.label for b in at.button])
    )


def test_a_finished_auditor_produces_a_report_on_one_click(app, finished_auditor):
    at = app()
    # Variant A is still a skeleton here, so its notice is expected.
    assert not any("Variant B" in m.value for m in at.info)
    review_button(at).click().run()
    assert not at.exception, at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "Prolonged execution" in body


def test_a_report_shows_its_limitations_uncollapsed(app, finished_auditor):
    """A report that hides what it could not establish fails this lab's point."""
    at = app()
    review_button(at).click().run()
    body = " ".join(m.value for m in at.markdown)
    assert "What this review could not establish" in body
    assert "Foot pedal activation was not visible." in body


def test_a_report_names_the_evidence_under_each_finding(app, finished_auditor):
    at = app()
    review_button(at).click().run()
    captions = " ".join(c.value for c in at.caption)
    assert "Evidence: 23.7 min against 16.5 min" in captions


# --------------------------------------------------------------------------
# preflight's Antigravity check — the IDE the session assumes
# --------------------------------------------------------------------------


def test_the_required_extensions_are_all_on_open_vsx():
    """Pins the three ids, and pins why Pylance is not among them.

    Antigravity installs from Open VSX only. `ms-python.vscode-pylance` is
    licensed to official VS Code and returns 404 there, so recommending it
    produces a prompt twenty-five people cannot satisfy. Verified against
    open-vsx.org on 2026-09-02: python 200, debugpy 200, pylance 404.
    """
    import preflight

    required = set(preflight.ANTIGRAVITY_EXTENSIONS)
    assert required == {
        "ms-python.python",
        "ms-python.debugpy",
        "detachhead.basedpyright",
    }
    assert "ms-python.vscode-pylance" not in required
    # Each id carries why it is needed; the warning text splices them in, so an
    # empty one produces a message that tells the reader nothing.
    assert all(preflight.ANTIGRAVITY_EXTENSIONS.values())


def test_the_extension_recommendations_match_what_preflight_checks():
    """A recommendation the check does not verify, or vice versa, drifts."""
    import json
    import re

    import preflight

    raw = (REPO_ROOT / ".vscode" / "extensions.json").read_text()
    cfg = json.loads(re.sub(r"^\s*//.*$", "", raw, flags=re.M))
    assert set(cfg["recommendations"]) == set(preflight.ANTIGRAVITY_EXTENSIONS)
    assert "ms-python.vscode-pylance" in cfg["unwantedRecommendations"]


def test_a_missing_ide_is_a_warning_not_a_failure(monkeypatch, tmp_path):
    """Everything in this lab runs from a terminal, so no IDE cannot fail."""
    import preflight

    monkeypatch.setattr(preflight.Path, "home", staticmethod(lambda: tmp_path))
    assert preflight._antigravity_extension_ids() is None


def test_extensions_are_read_from_the_manifest_and_the_directory(tmp_path, monkeypatch):
    """Both routes, because a half-written manifest is a real state."""
    import json

    import preflight

    root = tmp_path / ".antigravity-ide" / "extensions"
    root.mkdir(parents=True)
    monkeypatch.setattr(preflight.Path, "home", staticmethod(lambda: tmp_path))

    # directory-name fallback: publisher.name-version-platform
    (root / "ms-python.python-2026.4.0-universal").mkdir()
    assert preflight._antigravity_extension_ids() == {"ms-python.python"}

    # the manifest wins when it parses
    (root / "extensions.json").write_text(
        json.dumps([{"identifier": {"id": "detachhead.basedpyright"}}])
    )
    assert preflight._antigravity_extension_ids() == {"detachhead.basedpyright"}

    # and a corrupt manifest falls back rather than crashing
    (root / "extensions.json").write_text("{ not json")
    assert preflight._antigravity_extension_ids() == {"ms-python.python"}


# --------------------------------------------------------------------------
# sessions without footage — 149 of the 155, so participants will land on one
# --------------------------------------------------------------------------


def test_a_session_without_clips_still_shows_its_flagged_moments(app):
    """Detection needs no footage, so Lab 1 must work on all 155 sessions.

    Drives the real picker rather than the default, which is a curated
    session and would make this pass without touching the case it names.
    """
    at = app(**{"session_pick": "case_001"})
    assert not at.exception, at.exception
    assert at.session_state["session_pick"] == "case_001"

    flags = next(m for m in at.metric if m.label == "Flagged moments")
    assert int(flags.value) > 0, "case_001 has flags; Lab 1 does not need footage"

    # and the moment says why it cannot be explained, rather than offering to
    labels = " ".join(e.label for e in at.expander)
    assert "no clip available" in labels
    assert not any("Explain this moment" in b.label for b in at.button)


def test_an_unprepared_session_says_so_and_offers_no_button(app):
    """The failure a participant would otherwise find by pressing a button.

    case_001 has flags and no clips. The old behaviour offered "Explain this
    moment", then answered with a ClipUnavailable naming five clip ids from a
    different session, in red — which reads as a bug rather than a choice.
    """
    from lab.clips import find_for_window
    from lab.rules import find_deviations

    devs = find_deviations("case_001")
    assert devs, "case_001 should still produce flags"
    assert all(
        find_for_window(d.case_id, d.part, *d.watch_window) is None for d in devs
    ), "case_001 is the unprepared example; it must have no clips"


def test_a_prepared_session_does_have_clips(app):
    """The other half of the same claim, so the check cannot pass vacuously."""
    from lab.clips import find_for_window
    from lab.rules import find_deviations

    devs = find_deviations("case_129")
    assert devs
    assert all(
        find_for_window(d.case_id, d.part, *d.watch_window) is not None for d in devs
    ), "case_129 is curated; every flag should have footage"


def _timeline_spec(case_id):
    """Build the timeline chart without running the app's main()."""
    import pathlib
    import types

    from lab.data import load_case
    from lab.metrics import step_metrics
    from lab.rules import find_deviations

    src = pathlib.Path(APP).read_text().replace("\nmain()\n", "\n")
    mod = types.ModuleType("appmod")
    mod.__file__ = APP
    exec(compile(src, APP, "exec"), mod.__dict__)

    case = load_case(case_id)
    chart = mod.build_timeline(
        case, step_metrics(case), find_deviations(case_id), case.parts[0]
    )
    return chart.to_dict()


def test_the_timeline_gives_each_layer_its_own_colour_scale():
    """The bug that made the chart look empty on any session with a flag.

    Three layers encode colour on different fields: task, tool, rule. Altair
    layers share scales by default, and the flags layer pins an explicit
    domain of the four rule ids. Shared, that domain is imposed on the other
    two, every task and instrument value falls outside it, and the bands and
    arm bars render without colour.

    It was invisible in the obvious place to look, because a session with no
    flags has no third layer and therefore no pinned domain, and drew
    perfectly.
    """
    spec = _timeline_spec("case_045")
    assert spec["resolve"]["scale"]["color"] == "independent"

    fields = [L.get("encoding", {}).get("color", {}).get("field") for L in spec["layer"]]
    assert fields == ["task", "tool", "rule"], fields


def test_a_session_with_flags_still_carries_its_band_and_arm_data():
    """The symptom, checked directly: the marks exist and have rows."""
    spec = _timeline_spec("case_001")
    datasets = spec["datasets"]
    rows = [len(datasets[L["data"]["name"]]) for L in spec["layer"]]
    assert all(n > 0 for n in rows), f"a layer rendered with no rows: {rows}"
    assert len(spec["layer"]) == 3, "case_001 has flags, so it should have three layers"


# --------------------------------------------------------------------------
# lab/trace.py — making the pipeline visible while it runs
# --------------------------------------------------------------------------


def test_steps_go_nowhere_when_nobody_is_listening():
    """Scripts and tests call the same code; it must cost nothing there."""
    from lab import trace

    trace.step("this should vanish")
    assert trace._listeners == []


def test_a_listener_receives_steps_in_order_and_is_removed_after():
    from lab import trace

    seen = []
    with trace.listening(seen.append):
        trace.step("first")
        trace.step("second")
    trace.step("after")
    assert seen == ["first", "second"]
    assert trace._listeners == []


def test_a_broken_listener_cannot_break_the_pipeline_it_watches():
    """The display is a bystander. It must never take down the work."""
    from lab import trace

    def explode(_):
        raise RuntimeError("the UI fell over")

    seen = []
    with trace.listening(explode), trace.listening(seen.append):
        trace.step("still delivered to the good listener")
    assert seen == ["still delivered to the good listener"]


def test_the_listener_is_removed_even_if_the_body_raises():
    from lab import trace

    with contextlib.suppress(ValueError):
        with trace.listening(lambda m: None):
            raise ValueError("boom")
    assert trace._listeners == []


def test_the_supplied_pipeline_reports_its_own_steps():
    """Clip resolution and the cache announce themselves, so a participant
    who writes no trace calls still sees the surrounding work."""
    import inspect

    from lab import cache, clips

    assert "trace.step" in inspect.getsource(clips.resolve_clip)
    assert "trace.step" in inspect.getsource(cache.disk_cached)
