"""Lab 2 — ask Gemini what a flagged moment looks like.

YOU WRITE THIS FILE.

Lab 1 hands you a list: timestamps, rule names, and the measurement behind
each. "Tool-swap churn at 137 seconds" names a statistic, not what the
trainee's hands were doing. Turn one flagged window into a description a
person can act on, then refuse the ones you cannot defend.

Five things decide whether this works.

1. **Window it.** Send the 40-second ``watch_window`` each flag carries, not
   the whole step. About 10,000 tokens against 170,000, and it is what keeps
   twenty-five people inside one rate limit.

2. **Give the model the flag.** Not "what is wrong here" — tell it what the
   rules found and ask what is visible around that moment. The difference in
   output quality is not close.

3. **Say which clock you want.** The clip arrives with offsets, so the
   footage the model sees starts near zero while the window you want quoted is
   thousands of seconds into the session. Asked only to "cite timestamps" it
   answers in clip-offset seconds — reasonably — and your guardrail rejects it
   for a question it could not answer. Two thirds of first-attempt rejections,
   measured, until the reference prompt named the range.

4. **Constrain the shape.** The schema below is given. ``not_visible`` must
   come back non-empty: models overclaim on medical-adjacent video, and making
   honesty a required field is the fix.

5. **Check it before anyone sees it.** The schema fixes the shape, not the
   content: nothing stops a cited timestamp the model was never shown.
   ``validate()`` is yours to write too, and when it fires, tighten the prompt
   and run again.

Two things are done for you. ``lab.clips.resolve_clip()`` gets the clip to the
model whichever key you hold — the Gemini API cannot read ``gs://``, uploads
live 48 hours, and files belong to one project. And results are cached to
disk, so iterating on a prompt is free.

Call ``lab.trace.step("...")`` as you go and the interface prints your steps
live. The supplied parts already report themselves; the middle is yours.

A reference implementation is in ``solutions/``. Reading it after you have
tried is a normal move.

"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """One thing visible in the footage, tied to a moment."""

    t: float = Field(description="seconds from the start of the video part")
    what: str = Field(description="what is visibly happening")
    technique_note: str = Field(description="the coaching observation")
    confidence: Literal["clear", "probable", "uncertain"]


class TechniqueNotes(BaseModel):
    """Structured commentary on one window of a session.

    Note the window is two floats rather than a tuple. A ``tuple[float, float]``
    serialises to ``prefixItems`` in JSON Schema, which the Gemini API rejects
    outright — the request fails before it is sent. The same applies to any
    field you add: stick to str, float, int, bool, Literal, lists of those, and
    nested models.
    """

    case_id: str
    window_start_s: float
    window_end_s: float
    summary: str = Field(description="one or two sentences")
    observations: list[Observation] = Field(description="two to five, each timestamped")
    visible_factors: list[str] = Field(
        description="what in frame might explain the flag"
    )
    not_visible: list[str] = Field(
        description="what this footage cannot establish. Must not be empty."
    )


class GuardrailViolation(ValueError):
    """The model returned something the lab will not pass on to a person."""


def validate(notes: TechniqueNotes, t_start: float, t_end: float) -> None:
    """Refuse output that cannot be defended.

    **You write this.** A schema guarantees the *shape* of what comes back, not
    that it is true. This is the second half: a guardrail that runs on every
    response before anything downstream sees it.

    Four things are worth enforcing, and all four are checkable without a
    human:

    * **Every observation's timestamp falls inside the window.** Models cite
      moments they were never shown. A timestamp outside ``[t_start, t_end]``
      is fabricated by construction.
    * **``not_visible`` is non-empty.** Required because models overclaim on
      medical-adjacent footage. An empty list means it did not consider the
      limits of what it saw.
    * **No clinical vocabulary.** This reviews a training exercise. Words like
      "diagnosis", "patient", "complication" or "injury" put the output outside
      the scope the lab is allowed to occupy.
    * **At least one observation.** Notes with none are not notes.

    Raise :class:`GuardrailViolation` with a message naming what failed, so the
    caller can retry with a tightened prompt — which is the actual lesson.

    Args:
        notes: what the model returned.
        t_start: the window start that was requested, in seconds.
        t_end: the window end that was requested, in seconds.

    Raises:
        GuardrailViolation: naming the specific check that failed.
    """
    raise NotImplementedError(
        "Lab 2: write the guardrail. See the docstring above for the four "
        "checks, and solutions/ for one way to do it."
    )


def analyze_clip(
    case_id: str,
    part: int = 1,
    t_start: float | None = None,
    t_end: float | None = None,
) -> TechniqueNotes:
    """Describe what is visible in one window of a recorded session.

    Use this to explain a flagged moment in plain language. The description is
    advisory: it says what the footage shows, never what the numbers already
    established.

    Args:
        case_id: the session identifier, e.g. ``"case_045"``.
        part: the video part the window belongs to. Time restarts at zero in
            each part, so a window is only meaningful alongside its part.
        t_start: window start in seconds within that part. Omit to analyse the
            whole clip, which is what the Auditor variant wants.
        t_end: window end in seconds within that part.
    """
    raise NotImplementedError(
        "Lab 2: build a Gemini call that returns TechniqueNotes, then run it "
        "through validate(). See the module docstring, and solutions/ if you "
        "get stuck."
    )
