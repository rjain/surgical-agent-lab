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
