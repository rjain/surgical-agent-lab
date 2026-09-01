"""Measurements over a session's task and tool labels.

SUPPLIED — you do not need to change this file.

Everything here is arithmetic over the labels. No model, no video, no
inference. These are the numbers :mod:`lab.rules` thresholds against, and the
numbers your agent must cite rather than invent.

A note on what is *not* here. An earlier draft of this lab promised a "pause
ratio". The labels record when an instrument was mounted and unmounted, not
whether it was moving, so idle time is not observable from this data. Rather
than ship a metric that looks quantitative and means nothing, it is left out.
If you want it, you need instrument kinematics, which SurgVU does not publish.
"""

from __future__ import annotations

import json
import statistics
from functools import lru_cache
from pathlib import Path

import pandas as pd

from lab.data import Case, list_cases, load_case, tool_changes_within

# Median duration per task step across the whole corpus, precomputed by
# tools/build_cohort.py so participants do not wait on 155 cases at import.
_COHORT_PATH = Path(__file__).with_name("cohort.json")


@lru_cache(maxsize=1)
def cohort_medians() -> dict[str, float]:
    """Median duration in seconds for each task step, across the corpus.

    Falls back to computing from whatever cases are available locally if the
    precomputed file is missing.
    """
    if _COHORT_PATH.exists():
        return json.loads(_COHORT_PATH.read_text())["median_duration_s"]

    buckets: dict[str, list[float]] = {}
    for case_id in list_cases():
        for row in load_case(case_id).tasks.itertuples():
            buckets.setdefault(row.task, []).append(row.duration_s)
    return {k: statistics.median(v) for k, v in buckets.items() if v}


def step_metrics(case: Case) -> pd.DataFrame:
    """One row of measurements per task segment.

    Columns:
        ``part``, ``start_s``, ``end_s``, ``task``, ``duration_s`` — carried
        through from the labels.
        ``cohort_median_s`` — median duration of this step across the corpus.
        ``duration_ratio`` — this segment against that median. 1.0 is typical,
        2.0 is twice as long as usual.
        ``tool_changes`` — instrument mounts beginning inside the segment.
        ``swaps_per_min`` — those changes per minute of segment.
        ``arms_active`` — distinct arms carrying an instrument during it.
        ``distinct_tools`` — distinct instruments seen during it.
        ``has_unknown_tool`` — whether an unidentified instrument was mounted.

    Args:
        case: a loaded case, from :func:`lab.data.load_case`.
    """
    medians = cohort_medians()
    rows = []
    for seg in case.tasks.itertuples():
        changes = tool_changes_within(case.tools, seg.part, seg.start_s, seg.end_s)
        during = case.tools[
            (case.tools["part"] == seg.part)
            & (case.tools["start_s"] < seg.end_s)
            & (case.tools["end_s"] > seg.start_s)
        ]
        median = medians.get(seg.task)
        minutes = seg.duration_s / 60.0
        rows.append(
            {
                "part": seg.part,
                "start_s": seg.start_s,
                "end_s": seg.end_s,
                "task": seg.task,
                "duration_s": seg.duration_s,
                "cohort_median_s": median,
                "duration_ratio": (seg.duration_s / median) if median else None,
                "tool_changes": len(changes),
                "swaps_per_min": (len(changes) / minutes) if minutes else 0.0,
                "arms_active": during["arm"].nunique(),
                "distinct_tools": during["tool"].nunique(),
                "has_unknown_tool": bool((during["tool"] == "<unknown>").any()),
            }
        )
    return pd.DataFrame(rows)


def session_summary(case: Case) -> dict:
    """Headline numbers for a whole session.

    Args:
        case: a loaded case, from :func:`lab.data.load_case`.
    """
    metrics = step_metrics(case)
    labelled_s = float(case.tasks["duration_s"].sum())
    span_s = 0.0
    for part in case.parts:
        segments = case.tasks[case.tasks["part"] == part]
        if len(segments):
            span_s += float(segments["end_s"].max() - segments["start_s"].min())

    ratios = metrics["duration_ratio"].dropna()
    return {
        "case_id": case.case_id,
        "parts": case.parts,
        "segments": int(len(case.tasks)),
        "tool_mounts": int(len(case.tools)),
        "labelled_s": labelled_s,
        "span_s": span_s,
        "labelled_fraction": (labelled_s / span_s) if span_s else None,
        "distinct_tasks": int(case.tasks["task"].nunique()),
        "distinct_tools": int(case.tools["tool"].nunique()),
        "total_tool_changes": int(metrics["tool_changes"].sum()),
        "slowest_step": (
            metrics.loc[ratios.idxmax(), "task"] if len(ratios) else None
        ),
        "slowest_step_ratio": (float(ratios.max()) if len(ratios) else None),
    }


def get_metrics(case_id: str, step: str | None = None) -> dict:
    """Timing and tool-usage measurements for a session, or one step of it.

    Use this to answer questions about how long something took or how much
    instrument swapping happened. Every number returned is measured from the
    session labels — none of it is estimated.

    Args:
        case_id: the session identifier, e.g. ``"case_045"``.
        step: optional task step name, e.g. ``"Suturing"``. When given, only
            segments of that step are returned. When omitted, the whole
            session is summarised.
    """
    case = load_case(case_id)
    if step is None:
        return session_summary(case)

    metrics = step_metrics(case)
    matching = metrics[metrics["task"].str.lower() == step.lower()]
    if matching.empty:
        available = sorted(metrics["task"].unique())
        return {"error": f"no step named {step!r}", "available_steps": available}
    return {
        "case_id": case_id,
        "step": step,
        "segments": matching.to_dict("records"),
    }
