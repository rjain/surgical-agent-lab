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

from lab import opi
from lab.data import (
    Case,
    list_cases,
    load_case,
    overlapping_tools,
    tool_changes_within,
)

# Median duration per task step across the whole corpus, precomputed by
# the instructor pipeline so participants do not wait on 155 cases at import.
_COHORT_PATH = Path(__file__).with_name("cohort.json")



#: Which fields this lab computes correspond to a console metric, and which
#: are the lab's own. The near miss is deliberate: the console's
#: ``arm_swap_freq`` is the surgeon changing which arm the hand controllers
#: drive, not an instrument being exchanged, so ``swaps_per_min`` is not it.
_CONSOLE_FIELDS = {
    "duration_s": "duration",
    "duration_tool": "duration_tool",
    "duration_armxtool": "duration_armxtool",
}

_LAB_ONLY_NOTE = "not a console metric"


def _glossary(fields) -> dict[str, dict]:
    """Explain each returned field, in the console's words where there are any.

    Args:
        fields: the field names present in the payload.
    """
    out = {}
    for field in fields:
        console_name = _CONSOLE_FIELDS.get(field)
        entry = opi.describe(console_name) if console_name else None
        if entry:
            out[field] = {
                "console_name": entry["metric_name"],
                "display_name": entry["display_name"],
                "definition": entry["description"],
                "source": "console",
            }
        else:
            out[field] = {"source": "lab", "note": _LAB_ONLY_NOTE}
    return out


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


def _merged_seconds(spans: list[tuple[float, float]]) -> float:
    """Total time covered by a set of spans, counting overlap once.

    Args:
        spans: ``(start, end)`` pairs in seconds, in any order.
    """
    total = 0.0
    current_start = current_end = None
    for start, end in sorted(spans):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


def install_durations(
    case: Case, part: int, start_s: float, end_s: float
) -> dict[str, dict[str, float]]:
    """How long each instrument was installed during a window.

    These are the console's ``duration_tool`` and ``duration_armxtool``, two
    of the three OPI metrics SurgVU labels can support. Spans are clipped to
    the window, and overlapping mounts of the same instrument on the same arm
    are counted once: the labels occasionally record two mounts of one
    instrument on one arm at the same time, and adding them would credit an
    arm with more time than the window contains.

    Args:
        case: a loaded case, from :func:`lab.data.load_case`.
        part: the video part the window belongs to.
        start_s: window start, seconds within that part.
        end_s: window end, seconds within that part.

    Returns:
        ``{"duration_tool": {tool: seconds}, "duration_armxtool":
        {"USM3 needle driver": seconds}}``.
    """
    during = overlapping_tools(case.tools, part, start_s, end_s)

    per_arm_spans: dict[str, list[tuple[float, float]]] = {}
    for row in during.itertuples():
        span = (max(row.start_s, start_s), min(row.end_s, end_s))
        per_arm_spans.setdefault(f"{row.arm} {row.tool}", []).append(span)

    per_arm = {key: _merged_seconds(spans) for key, spans in per_arm_spans.items()}

    per_tool: dict[str, float] = {}
    for key, seconds in per_arm.items():
        tool = key.split(" ", 1)[1]
        per_tool[tool] = per_tool.get(tool, 0.0) + seconds

    return {"duration_tool": per_tool, "duration_armxtool": per_arm}


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


def get_metrics(
    case_id: str, step: str | None = None, metric: str | None = None
) -> dict:
    """Timing and tool-usage measurements for a session, or one step of it.

    Use this to answer questions about how long something took or how much
    instrument swapping happened. Every number returned is measured from the
    session labels — none of it is estimated.

    The reply carries a ``glossary`` explaining each field, and giving the
    console's own name and definition for the fields that have one.

    Args:
        case_id: the session identifier, e.g. ``"case_045"``.
        step: optional task step name, e.g. ``"Suturing"``. When given, only
            segments of that step are returned. When omitted, the whole
            session is summarised.
        metric: optional console metric name, e.g. ``"duration_armxtool"``.
            Most console metrics cannot be computed from these labels; asking
            for one of those returns an explanation and no number.
    """
    if metric is not None:
        entry = opi.describe(metric)
        if entry is None:
            return {
                "unknown_metric": metric,
                "hint": "the console does not publish a metric by that name",
            }
        if not entry["derivable"]:
            return {
                "unavailable": metric,
                "display_name": entry["display_name"],
                "definition": entry["description"],
                "reason": (
                    "requires console telemetry; these labels record instrument "
                    "mounting, not motion, force or pedal use"
                ),
            }

    case = load_case(case_id)

    if step is None:
        summary = session_summary(case)
        summary["glossary"] = _glossary(summary)
        return summary

    metrics = step_metrics(case)
    matching = metrics[metrics["task"].str.lower() == step.lower()]
    if matching.empty:
        available = sorted(metrics["task"].unique())
        return {"error": f"no step named {step!r}", "available_steps": available}

    segments = matching.to_dict("records")
    for record in segments:
        record.update(
            install_durations(case, record["part"], record["start_s"], record["end_s"])
        )

    payload = {"case_id": case_id, "step": step, "segments": segments}
    if metric is not None:
        payload["metric"] = metric
    payload["glossary"] = _glossary(segments[0])
    return payload
