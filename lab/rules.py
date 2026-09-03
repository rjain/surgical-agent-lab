"""The deterministic deviation engine.

SUPPLIED — you do not need to change this file, though tuning the thresholds
is a good stretch exercise.

This is the layer that decides what is worth a human's attention. It is
ordinary Python over measured numbers: microseconds to run, the same answer
every time, and reviewable line by line. **No model is involved, and none
should be.** Everything downstream is advisory; this is not.

Every flag carries the measurement that produced it, so any claim made about
it later can be traced back to a number.

A note on the vocabulary. The field is called ``score`` and not ``severity``
or ``risk``. These are observations about the efficiency of a recorded
training exercise. They are not clinical judgements and must not be presented
as any.

**Why these rules and not others.** An obvious candidate — "an unexpected
instrument for this step" — does not survive contact with the corpus.
Monopolar curved scissors are mounted during 98% of Uterine horn segments and
58% of Rectal artery/vein segments. Flagging that combination flags the normal
case and teaches the room that the system cries wolf. The rules below were
each measured across all 155 cases first; see ``tools/evaluate_rules.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from lab.data import Case, load_case, tool_changes_within
from lab.metrics import step_metrics

# --- thresholds -------------------------------------------------------------
# Tuned against the five curated cases. Loosening OVERRUN_RATIO below about
# 1.5 is what makes that rule useful at all: at 1.75 it fires once across the
# whole sample set.
SWAP_CHANGES = 4          # instrument changes beginning inside one segment
OVERRUN_RATIO = 1.4       # segment duration against the corpus median
MIN_SEGMENT_S = 60.0      # ignore very short segments; ratios are noisy there

#: How much footage is worth watching around a flag, in seconds.
#: Flagged segments run from 4 to 45 minutes — a median of eleven — and sending
#: one whole to a model costs roughly 170,000 tokens. Every rule therefore
#: nominates a `focus_s`: the instant that actually explains the flag. Layer 2
#: looks at a window this wide around that instant, which costs about 5,000.
FOCUS_WINDOW_S = 40.0


@dataclass(frozen=True)
class Deviation:
    """One flagged moment.

    Attributes:
        case_id: the session it belongs to.
        part: video part, because time restarts in each one.
        start_s: window start, seconds within that part.
        end_s: window end, seconds within that part.
        step: the task step under way.
        rule_id: which rule fired.
        score: 0-1, how far past the threshold this is. An efficiency
            observation, not a clinical severity.
        evidence: the measurement that tripped the rule, in words.
    """

    case_id: str
    part: int
    start_s: float
    end_s: float
    step: str
    rule_id: str
    score: float
    evidence: str
    focus_s: float = 0.0

    def as_dict(self) -> dict:
        """Plain-dict form, for JSON and for passing to a model."""
        return asdict(self)

    @property
    def watch_window(self) -> tuple[float, float]:
        """The stretch of footage worth actually looking at.

        A flagged segment can run three quarters of an hour. This narrows it to
        :data:`FOCUS_WINDOW_S` seconds around ``focus_s``, clamped to the
        segment, which is what makes explaining a flag affordable.
        """
        half = FOCUS_WINDOW_S / 2
        centre = self.focus_s or (self.start_s + self.end_s) / 2
        lo = max(self.start_s, centre - half)
        hi = min(self.end_s, lo + FOCUS_WINDOW_S)
        return (lo, max(hi, lo + 1.0))


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _unknown_mount_time(case: Case, row) -> float:
    """When the unidentified instrument went on, which is the moment to watch."""
    during = case.tools[
        (case.tools["part"] == int(row.part))
        & (case.tools["start_s"] < row.end_s)
        & (case.tools["end_s"] > row.start_s)
        & (case.tools["tool"] == "<unknown>")
    ]
    if not len(during):
        return 0.0
    return float(max(during["start_s"].min(), row.start_s))


def swap_rate_outliers(case: Case, metrics: pd.DataFrame) -> list[Deviation]:
    """Segments with an unusual burst of instrument changes.

    Most segments contain no instrument changes at all — swaps normally happen
    in the unlabelled gaps between steps. A cluster inside one step is a real
    signal of hesitation or of a plan being revised mid-step.
    """
    out = []
    for row in metrics.itertuples():
        if row.tool_changes < SWAP_CHANGES:
            continue
        # the densest cluster of changes is what a reviewer wants to see
        changes = tool_changes_within(
            case.tools, int(row.part), row.start_s, row.end_s
        )
        focus = float(changes["start_s"].median()) if len(changes) else 0.0
        out.append(
            Deviation(
                case_id=case.case_id,
                part=int(row.part),
                start_s=float(row.start_s),
                end_s=float(row.end_s),
                step=row.task,
                rule_id="swap_rate",
                score=_clamp((row.tool_changes - SWAP_CHANGES + 1) / 6.0),
                evidence=(
                    f"{row.tool_changes} instrument changes in "
                    f"{row.duration_s / 60:.1f} min "
                    f"({row.swaps_per_min:.2f}/min)"
                ),
                focus_s=focus,
            )
        )
    return out


def step_overruns(case: Case, metrics: pd.DataFrame) -> list[Deviation]:
    """Segments that ran long against the corpus median for that step.

    The comparison is per step type: a 10-minute Uterine horn is quick, a
    10-minute Retraction is not.
    """
    out = []
    for row in metrics.itertuples():
        if row.duration_ratio is None or pd.isna(row.duration_ratio):
            continue
        if row.duration_s < MIN_SEGMENT_S or row.duration_ratio < OVERRUN_RATIO:
            continue
        out.append(
            Deviation(
                case_id=case.case_id,
                part=int(row.part),
                start_s=float(row.start_s),
                end_s=float(row.end_s),
                step=row.task,
                rule_id="step_overrun",
                score=_clamp((row.duration_ratio - OVERRUN_RATIO) / 1.5),
                evidence=(
                    f"{row.duration_s / 60:.1f} min against a corpus median of "
                    f"{row.cohort_median_s / 60:.1f} min "
                    f"({row.duration_ratio:.2f}x)"
                ),
                # nothing single-instant about running long, so look at the
                # point the step passed its expected duration
                focus_s=float(row.start_s) + float(row.cohort_median_s or 0.0),
            )
        )
    return out


def unknown_instruments(case: Case, metrics: pd.DataFrame) -> list[Deviation]:
    """Segments during which an unidentified instrument was mounted.

    The log records that something was installed on an arm but not what. Rare,
    and worth surfacing: any downstream reasoning about that window is working
    with an incomplete picture.
    """
    out = []
    for row in metrics.itertuples():
        if not row.has_unknown_tool:
            continue
        out.append(
            Deviation(
                case_id=case.case_id,
                part=int(row.part),
                start_s=float(row.start_s),
                end_s=float(row.end_s),
                step=row.task,
                rule_id="unknown_instrument",
                score=0.4,
                evidence="an instrument was mounted but not identified in the log",
                focus_s=_unknown_mount_time(case, row),
            )
        )
    return out


def step_oscillations(case: Case, metrics: pd.DataFrame) -> list[Deviation]:
    """A step returned to after a different step ran in between.

    An A - B - A pattern means the first attempt at A did not finish the job.
    The flag is placed on the second A, which is the part worth reviewing.
    """
    out = []
    tasks = list(metrics.itertuples())
    for i in range(2, len(tasks)):
        first, middle, repeat = tasks[i - 2], tasks[i - 1], tasks[i]
        if first.task != repeat.task or first.task == middle.task:
            continue
        if first.part != repeat.part:
            continue
        gap_min = (repeat.start_s - first.end_s) / 60
        out.append(
            Deviation(
                case_id=case.case_id,
                part=int(repeat.part),
                start_s=float(repeat.start_s),
                end_s=float(repeat.end_s),
                step=repeat.task,
                rule_id="step_oscillation",
                score=0.5,
                evidence=(
                    # Both numbers, because they are very different and the
                    # second one alone reads stronger than it is. The middle
                    # step's own duration says nothing about how much time
                    # actually passed: these labels are islands, covering
                    # under a fifth of most sessions, so an A-B-A can span
                    # more than an hour of unlabelled work.
                    f"returned to {repeat.task!r} {gap_min:.1f} min later; "
                    f"{middle.task!r} ({middle.duration_s / 60:.1f} min) was "
                    "the only step labelled in between"
                ),
                # the start of the second attempt is the interesting part
                focus_s=float(repeat.start_s) + FOCUS_WINDOW_S / 2,
            )
        )
    return out


#: Every rule the engine runs, in the order results are reported.
#: What each rule is looking for, in words, and the threshold it uses.
#:
#: The measurement in ``evidence`` says what happened. It does not say why
#: anyone should care, and "returned to Suturing" reads as unremarkable until
#: you know the rule was looking for exactly that. Both halves are needed, so
#: the interface prints this above the evidence.
#:
#: The thresholds are interpolated from the constants above rather than
#: retyped, so they cannot drift and there is nothing here for a test to
#: guard. Two attempts at one were tautological before that sank in.
RULE_INTENT = {
    "swap_rate": (
        "A step with an unusual amount of instrument changing. Swapping is "
        f"normal; {SWAP_CHANGES} or more changes beginning inside one step "
        "often means the plan for that step changed while it was under way."
    ),
    "step_overrun": (
        "A step that took substantially longer than the same step usually "
        f"takes across the corpus. Fires past {OVERRUN_RATIO:g}x the median, "
        "so it is a comparison against peers rather than against a target."
    ),
    "step_oscillation": (
        "A step that was returned to after a different step ran in between. "
        "Going back suggests the first attempt did not finish the job, and "
        "the footage either side is usually where the reason shows. Weigh it "
        "by the elapsed gap: these labels cover under a fifth of a session, "
        "so a long gap means plenty happened that nobody labelled and the "
        "return may not be a return at all."
    ),
    "unknown_instrument": (
        "An instrument was mounted but the logs do not identify it. Nothing "
        "is necessarily wrong; it means the record is incomplete, so any "
        "count of what was used that step is a lower bound."
    ),
}


def rule_intent(rule_id: str) -> str:
    """What a rule looks for, in words. Empty string if the rule is unknown."""
    return RULE_INTENT.get(rule_id, "")


RULES = (
    swap_rate_outliers,
    step_overruns,
    step_oscillations,
    unknown_instruments,
)


def find_deviations(case_id: str) -> list[Deviation]:
    """Run every rule over one session.

    Args:
        case_id: the session identifier, e.g. ``"case_045"``.
    """
    case = load_case(case_id)
    metrics = step_metrics(case)
    found: list[Deviation] = []
    for rule in RULES:
        found.extend(rule(case, metrics))
    return sorted(found, key=lambda d: (d.part, d.start_s, d.rule_id))


def list_deviations(case_id: str) -> list[dict]:
    """Flagged moments in a recorded session, worth a reviewer's attention.

    Each entry gives the task step, which rule fired, the measurement behind
    it, and the stretch of footage worth watching. Pass ``watch_start_s`` and
    ``watch_end_s`` straight to the clip analyser to see what happened — the
    full ``start_s`` to ``end_s`` span is the whole task step and far too long
    to look at.

    These are efficiency observations about a recorded training exercise, not
    clinical findings.

    Args:
        case_id: the session identifier, e.g. ``"case_045"``.
    """
    out = []
    for dev in find_deviations(case_id):
        entry = dev.as_dict()
        watch_start, watch_end = dev.watch_window
        entry["watch_start_s"] = round(watch_start, 1)
        entry["watch_end_s"] = round(watch_end, 1)
        out.append(entry)
    return out
