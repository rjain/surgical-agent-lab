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
from lab.rules import find_deviations  # noqa: E402

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
            st.write(f"**Evidence.** {dev.evidence}")
            st.caption(f"score {dev.score} · an efficiency observation, not a clinical finding")
            during = overlapping_tools(case.tools, dev.part, dev.start_s, dev.end_s)
            st.dataframe(
                during[["arm", "tool", "commercial", "start_s", "end_s"]],
                width="stretch",
                hide_index=True,
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
                            notes = analyze(dev.case_id, dev.start_s, dev.end_s)
                            render_notes(notes)
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


def render_coach(case_id: str) -> None:
    build_coach, build_tracker = _load_coach()
    if build_coach is None:
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
    st.success("Coach loaded. Wire the runner in to start the conversation.")


def render_auditor(case_id: str) -> None:
    build_auditor = _load_auditor()
    if build_auditor is None:
        st.info(
            "**Variant B is not wired up yet.** Write `build_auditor()` in "
            "`lab/variants/auditor.py` and this tab runs it over the session."
        )
        return
    st.success("Auditor loaded. Wire the runner in to produce a report.")


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
