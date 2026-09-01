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

from lab.data import Case, load_case
from lab.metrics import step_metrics

# --- thresholds -------------------------------------------------------------
# Tuned against the five curated cases. Loosening OVERRUN_RATIO below about
# 1.5 is what makes that rule useful at all: at 1.75 it fires once across the
# whole sample set.
SWAP_CHANGES = 4          # instrument changes beginning inside one segment
OVERRUN_RATIO = 1.4       # segment duration against the corpus median
MIN_SEGMENT_S = 60.0      # ignore very short segments; ratios are noisy there


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

    def as_dict(self) -> dict:
        """Plain-dict form, for JSON and for passing to a model."""
        return asdict(self)


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


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
                    f"returned to {repeat.task!r} after {middle.task!r} "
                    f"ran for {middle.duration_s / 60:.1f} min in between"
                ),
            )
        )
    return out


#: Every rule the engine runs, in the order results are reported.
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

    Each entry gives the step, the time window, which rule fired, and the
    measurement behind it. These are efficiency observations about a training
    exercise, not clinical findings.

    Args:
        case_id: the session identifier, e.g. ``"case_045"``.
    """
    return [d.as_dict() for d in find_deviations(case_id)]
