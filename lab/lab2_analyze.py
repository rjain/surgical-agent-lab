"""Lab 2 — ask Gemini what a flagged moment looks like.

YOU WRITE THIS FILE.

Lab 1 gives you a list: timestamps, rule names, and the measurement behind
each. That is enough for someone who already knows the exercise, and close to
useless for the trainee who does not. "Tool-swap churn at 137 seconds" names a
statistic, not what their hands were doing.

Your job is to turn one flagged window into a description a person can act on:
parseable, anchored to timestamps, and honest about what the video cannot
establish.

Three things decide whether this works:

1. **Window, do not send the whole clip.** A twenty-second window costs roughly
   5,500 tokens; a three-minute clip costs roughly 50,000, for less relevant
   footage. Pass the offsets through to the video part.
2. **Give the model the flag.** Do not ask the open question "what is wrong
   here". Tell it what the rules found and ask what is visible around that
   moment. The difference in output quality is not close.
3. **Constrain the output with the schema below.** Free text cannot be built
   on. `not_visible` must come back non-empty — models overclaim on
   medical-adjacent video, and making honesty a required field is the fix.
4. **Check the output before anyone sees it.** The schema fixes the shape; it
   does not stop the model citing a timestamp it was never shown. `validate()`
   is the guardrail, and you write it too. When it fails, tighten the prompt
   and run again — that loop is the point of this lab.

Two mechanics worth knowing before you start, both measured against the real
API rather than guessed:

* **Getting the clip to the model is done for you.** Call
  `lab.clips.resolve_clip(clip_id)` and you get a Files API URI that works with
  whichever key you are using. The Gemini API cannot read `gs://`, uploads live
  only 48 hours, and pre-uploaded files belong to the project their key came
  from — `resolve_clip` handles all three and falls back to uploading a local
  copy when it has to.
* **Window the video, and use the window the rules give you.** Flagged segments
  run from 4 to 45 minutes; sending one whole costs about 170,000 tokens. Every
  `Deviation` carries a `watch_window` — 40 seconds around the instant that
  explains the flag — which costs about 10,000. That is a 16-fold saving and it
  is what keeps twenty-five people inside a shared rate limit.

A reference implementation is in `solutions/`. Using it is a normal move, not
a defeat — but write your own prompt first.
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
