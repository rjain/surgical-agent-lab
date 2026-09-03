"""The lab interface.

SUPPLIED — you do not need to change this file, though you are welcome to.

Three tabs:

* **Session** — works from the first minute. A picture of the label files: the
  timeline, the flagged moments, and the measurements behind them. No video and
  nothing live — `rules.py` ran over the whole session before the page rendered.
* **Coach** — comes alive when you finish Variant A.
* **Auditor** — comes alive when you finish Variant B.

Run it with::

    streamlit run ui/app.py

The Explain action on each flagged moment calls *your* ``analyze_clip`` from
``lab/lab2_analyze.py``. Until you write it, it reports that it is not wired up
yet — that is expected, not a bug.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.data import list_cases, load_case, overlapping_tools  # noqa: E402
from lab.metrics import session_summary, step_metrics  # noqa: E402
from lab.rules import find_deviations, rule_intent  # noqa: E402

# Prepared for the lab and verified to contain flags worth discussing. Shown
# first so nobody's first impression is a session with nothing in it.
CURATED = ["case_045", "case_129", "case_125", "case_036", "case_044", "case_059"]

RULE_COLOURS = {
    "swap_rate": "#0F5F63",
    "step_overrun": "#B4611E",
    "step_oscillation": "#4A6FA5",
    "unknown_instrument": "#7A4F9E",
}

st.set_page_config(page_title="Surgical Workflow Deviation Auditor", layout="wide")


# --------------------------------------------------------------------------
# optional participant code — absent until they write it
# --------------------------------------------------------------------------


def _load_analyzer():
    try:
        from lab.lab2_analyze import analyze_clip

        return analyze_clip
    except Exception:
        return None


def _load_coach():
    try:
        from lab.variants.coach import build_coach, build_workflow_tracker

        return build_coach, build_workflow_tracker
    except Exception:
        return None, None


def _load_auditor():
    try:
        from lab.variants.auditor import build_auditor

        return build_auditor
    except Exception:
        return None


# --------------------------------------------------------------------------
# timeline
# --------------------------------------------------------------------------


def build_timeline(case, metrics: pd.DataFrame, deviations, part: int):
    """Layered chart: step bands, one lane per arm, a mark per flag.

    This is a picture of the label files and nothing more. Every mark on it
    comes from `rules.py`, which ran over the whole session before the page
    rendered — there is no video here and nothing is happening live.
    """
    steps = metrics[metrics["part"] == part].copy()
    steps["start_m"] = steps["start_s"] / 60
    steps["end_m"] = steps["end_s"] / 60
    steps["lane"] = "task step"

    tools = case.tools[case.tools["part"] == part].copy()
    tools["start_m"] = tools["start_s"] / 60
    tools["end_m"] = tools["end_s"] / 60
    tools["lane"] = tools["arm"].astype(str)

    lanes = ["task step"] + sorted(tools["lane"].unique())

    band = (
        alt.Chart(steps)
        .mark_bar(height=18, cornerRadius=2)
        .encode(
            x=alt.X("start_m:Q", title="minutes into part"),
            x2="end_m:Q",
            y=alt.Y("lane:N", sort=lanes, title=None),
            color=alt.Color("task:N", legend=alt.Legend(title="task step")),
            tooltip=["task", "duration_s", "tool_changes", "duration_ratio"],
        )
    )
    arms = (
        alt.Chart(tools)
        .mark_bar(height=11, opacity=0.75, cornerRadius=1)
        .encode(
            x="start_m:Q",
            x2="end_m:Q",
            y=alt.Y("lane:N", sort=lanes),
            color=alt.Color("tool:N", legend=alt.Legend(title="instrument")),
            tooltip=["arm", "tool", "commercial", "duration_s"],
        )
    )
    layers = [band, arms]

    here = [d for d in deviations if d.part == part]
    if here:
        flags = pd.DataFrame(
            {
                "mid_m": [(d.start_s + d.end_s) / 120 for d in here],
                "rule": [d.rule_id for d in here],
                "evidence": [d.evidence for d in here],
                "step": [d.step for d in here],
                "lane": ["task step"] * len(here),
            }
        )
        layers.append(
            alt.Chart(flags)
            .mark_point(size=150, shape="triangle-down", filled=True, yOffset=-20)
            .encode(
                x="mid_m:Q",
                y=alt.Y("lane:N", sort=lanes),
                color=alt.Color(
                    "rule:N",
                    scale=alt.Scale(
                        domain=list(RULE_COLOURS), range=list(RULE_COLOURS.values())
                    ),
                    legend=alt.Legend(title="flagged"),
                ),
                tooltip=["rule", "step", "evidence"],
            )
        )

    return alt.layer(*layers).properties(height=28 * len(lanes) + 60)


# --------------------------------------------------------------------------
# session tab
# --------------------------------------------------------------------------


def render_session(case_id: str) -> None:
    case = load_case(case_id)
    metrics = step_metrics(case)
    deviations = find_deviations(case_id)
    summary = session_summary(case)

    cols = st.columns(5)
    cols[0].metric("Segments", summary["segments"])
    cols[1].metric("Flagged moments", len(deviations))
    cols[2].metric("Instrument mounts", summary["tool_mounts"])
    cols[3].metric("Labelled", f"{(summary['labelled_fraction'] or 0) * 100:.0f}%")
    cols[4].metric("Video parts", len(case.parts))

    if case.is_multipart:
        st.caption(
            f"This session is split across parts {case.parts}. Time restarts at "
            "zero in each part, so the parts are shown separately."
        )

    part = case.parts[0]
    if len(case.parts) > 1:
        part = st.radio("Part", case.parts, horizontal=True)

    in_part = metrics[metrics["part"] == part]
    part_end = float(in_part["end_s"].max()) if len(in_part) else 0.0
    part_start = float(in_part["start_s"].min()) if len(in_part) else 0.0

    st.altair_chart(
        build_timeline(case, metrics, deviations, part), width="stretch"
    )

    st.subheader("Flagged moments")
    here = [d for d in deviations if d.part == part]
    if not here:
        st.info("No flags in this part. That is a legitimate result, not a failure.")
    analyze = _load_analyzer()

    for i, dev in enumerate(here):
        window = f"{dev.start_s / 60:.1f}–{dev.end_s / 60:.1f} min"
        label = f"{dev.rule_id} — {dev.step} — {window}"
        with st.expander(label, expanded=False):
            # Two halves, and neither works alone: the intent says what the
            # rule was looking for, the evidence says what it found. Without
            # the first, "returned to Suturing" reads as unremarkable.
            if intent := rule_intent(dev.rule_id):
                st.write(f"**What this rule looks for.** {intent}")
            st.write(f"**What it found here.** {dev.evidence}")
            st.caption(f"score {dev.score} · an efficiency observation, not a clinical finding")
            during = overlapping_tools(case.tools, dev.part, dev.start_s, dev.end_s)
            st.dataframe(
                during[["arm", "tool", "commercial", "start_s", "end_s"]],
                width="stretch",
                hide_index=True,
            )
            lo, hi = dev.watch_window
            st.caption(
                f"Watch window {lo:.0f}–{hi:.0f}s — the 40 seconds around the "
                "instant that explains this flag, which is what gets sent."
            )
            if st.button("Explain this moment", key=f"explain_{part}_{i}"):
                if analyze is None:
                    st.warning(
                        "`lab/lab2_analyze.py` is not written yet — that is Lab 2. "
                        "Once `analyze_clip` exists, its notes appear here."
                    )
                else:
                    with st.spinner("Asking Gemini about this window…"):
                        try:
                            # part, then offsets within that part — the whole
                            # segment would be 45 minutes of video.
                            notes = analyze(dev.case_id, dev.part, lo, hi)
                            render_notes(notes)
                        except NotImplementedError:
                            st.warning(
                                "`analyze_clip` is still a skeleton — that is "
                                "Lab 2. Its notes appear here once you write it."
                            )
                        except Exception as exc:
                            st.error(f"{type(exc).__name__}: {exc}")

    with st.expander("All measurements for this part"):
        st.dataframe(in_part, width="stretch", hide_index=True)


def render_notes(notes) -> None:
    """Render TechniqueNotes as structure, so you can see your schema worked."""
    data = notes if isinstance(notes, dict) else getattr(notes, "model_dump", dict)()
    if summary := data.get("summary"):
        st.write(summary)
    for obs in data.get("observations", []):
        left, right = st.columns([1, 6])
        left.code(f"{obs.get('t', 0):.1f}s")
        right.write(
            f"**{obs.get('what', '')}** — {obs.get('technique_note', '')} "
            f"`{obs.get('confidence', '')}`"
        )
    if factors := data.get("visible_factors"):
        st.caption("Visible factors: " + "; ".join(factors))
    if not_visible := data.get("not_visible"):
        with st.expander("What this footage can't show"):
            for item in not_visible:
                st.write(f"- {item}")


# --------------------------------------------------------------------------
# variant tabs
# --------------------------------------------------------------------------


def _coach_for(case_id: str):
    """Build the Coach once and keep it across reruns.

    Streamlit re-executes this file on every interaction, so a conversation
    built inline would forget the previous question. It is cached per session
    id, and switching sessions deliberately starts a fresh conversation.
    """
    from lab.runtime import Conversation

    build_coach, build_tracker = _load_coach()
    if build_coach is None:
        return None, None

    key = f"coach::{case_id}"
    if key not in st.session_state:
        # The skeleton defines these functions and raises from inside them, so
        # importing tells us nothing about whether they are written. Calling
        # them is the only test, and both outcomes are expected states:
        # no Tracker is the minimum shippable form, and no Coach means Variant
        # A has not been started.
        note = ""
        try:
            agent = build_coach(build_tracker())
        except NotImplementedError:
            try:
                agent = build_coach()
            except NotImplementedError:
                return None, None
            note = "single-agent form — WorkflowTracker not written yet"
        st.session_state[key] = (Conversation(agent), note)
        st.session_state[f"log::{case_id}"] = []
    return st.session_state[key]


def render_trace(calls) -> None:
    """Show what the agent actually consulted.

    The highest-value thing on this page. Prose reads the same whether a
    number came from a measurement or from the model's imagination, and this
    is the only way to tell the two apart from outside.
    """
    if not calls:
        st.caption(
            "⚠️ No tools called — this answer came from the model alone. "
            "Treat any number in it as invented."
        )
        return
    with st.expander(f"Grounding — {len(calls)} tool call(s)"):
        for call in calls:
            st.markdown(f"**`{call.name}`**")
            st.json(call.args, expanded=False)
            if call.response is not None:
                st.json(call.response, expanded=False)


def render_coach(case_id: str) -> None:
    conversation, note = _coach_for(case_id)
    if conversation is None:
        st.info(
            "**Variant A is not wired up yet.** Write `build_workflow_tracker()` "
            "and `build_coach()` in `lab/variants/coach.py` and this tab becomes "
            "a chat with your agent."
        )
        st.caption(
            "The trace of which tools were called appears under each answer. "
            "Without it you cannot tell grounding from invention."
        )
        return

    if note:
        st.caption(f"Coach loaded — {note}.")
    st.caption(
        f"Asking about **{case_id}**. Try: *which step ran longest?* · "
        "*what should I look at next?* · *what does the footage show there?*"
    )

    log_key = f"log::{case_id}"
    for turn in st.session_state.get(log_key, []):
        with st.chat_message(turn["role"]):
            st.markdown(turn["text"])
            if turn["role"] == "assistant":
                render_trace(turn.get("calls", []))

    question = st.chat_input("Ask about this session")
    if not question:
        return

    st.session_state[log_key].append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                # The agent is told which session without cluttering the
                # displayed question.
                reply = conversation.ask(f"About session {case_id}: {question}")
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                return
        st.markdown(reply.text or "_no answer returned_")
        render_trace(reply.calls)
    st.session_state[log_key].append(
        {"role": "assistant", "text": reply.text, "calls": reply.calls}
    )


def render_report(report) -> None:
    """Render a SessionReport, whatever shape the group gave it."""
    data = report if isinstance(report, dict) else getattr(report, "model_dump", dict)()

    st.subheader(data.get("headline", "Session report"))
    if vs := data.get("duration_vs_cohort"):
        st.caption(vs)

    findings = data.get("findings", [])
    st.markdown(f"**Findings** ({len(findings)})")
    for f in findings:
        colour = RULE_COLOURS.get(f.get("rule_id", ""), "#666")
        st.markdown(
            f"<span style='color:{colour}'>●</span> "
            f"**{f.get('rank', '?')}. {f.get('headline', '')}** — "
            f"{f.get('step', '')} @ {f.get('t', 0):.0f}s "
            f"<code>{f.get('rule_id', '')}</code>",
            unsafe_allow_html=True,
        )
        if detail := f.get("detail"):
            st.write(detail)
        if evidence := f.get("evidence"):
            st.caption(f"Evidence: {evidence}")

    if recs := data.get("recommendations"):
        st.markdown("**Recommendations**")
        for r in recs:
            st.write(f"- {r}")

    # Not collapsed. A report that hides what it could not establish is the
    # failure mode this whole lab is arranged against.
    if limits := data.get("limitations"):
        st.markdown("**What this review could not establish**")
        for item in limits:
            st.write(f"- {item}")

    with st.expander("Raw report"):
        st.json(data)


def render_auditor(case_id: str) -> None:
    build_auditor = _load_auditor()
    auditor = None
    if build_auditor is not None:
        try:
            auditor = build_auditor()
        except NotImplementedError:
            auditor = None
    if auditor is None:
        st.info(
            "**Variant B is not wired up yet.** Write `build_auditor()` in "
            "`lab/variants/auditor.py` and this tab runs it over the session."
        )
        return

    st.caption(
        "One action, unattended, over the whole session. Expect roughly one "
        "clip call per flagged moment, so give it half a minute."
    )
    key = f"report::{case_id}"
    if st.button(f"Review {case_id}", type="primary"):
        with st.spinner("Reviewing the session…"):
            try:
                st.session_state[key] = (
                    auditor(case_id) if callable(auditor) else auditor
                )
            except Exception as exc:
                st.error(f"{type(exc).__name__}: {exc}")
                return

    if key in st.session_state:
        render_report(st.session_state[key])


# --------------------------------------------------------------------------


def main() -> None:
    st.title("Surgical Workflow Deviation Auditor")
    st.caption(
        "Retrospective review of recorded **training exercise** footage. "
        "Not a medical device; not for clinical use."
    )

    cases = list_cases()
    if not cases:
        st.error(
            "No cases found. Set `LAB_DATA_DIR` to the folder holding the "
            "`case_*` directories, then reload."
        )
        return

    prepared = [c for c in CURATED if c in cases]
    others = [c for c in cases if c not in prepared]
    ordered = prepared + others
    case_id = st.sidebar.selectbox(
        "Session",
        ordered,
        index=0,
        format_func=lambda c: f"{c}  ·  prepared" if c in prepared else c,
    )
    st.sidebar.caption(
        f"{len(prepared)} prepared for the lab, {len(others)} others available"
    )

    session, coach, auditor = st.tabs(["Session", "Coach", "Auditor"])
    with session:
        render_session(case_id)
    with coach:
        render_coach(case_id)
    with auditor:
        render_auditor(case_id)


main()
