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
    """Structured commentary on one window of a session."""

    case_id: str
    window: tuple[float, float]
    summary: str = Field(description="one or two sentences")
    observations: list[Observation] = Field(description="two to five, each timestamped")
    visible_factors: list[str] = Field(
        description="what in frame might explain the flag"
    )
    not_visible: list[str] = Field(
        description="what this footage cannot establish. Must not be empty."
    )


def analyze_clip(
    case_id: str,
    t_start: float | None = None,
    t_end: float | None = None,
) -> TechniqueNotes:
    """Describe what is visible in one window of a recorded session.

    Use this to explain a flagged moment in plain language. The description is
    advisory: it says what the footage shows, never what the numbers already
    established.

    Args:
        case_id: the session identifier, e.g. ``"case_045"``.
        t_start: window start in seconds. Omit to analyse the whole clip,
            which is what the Auditor variant wants.
        t_end: window end in seconds.
    """
    raise NotImplementedError(
        "Lab 2: build a Gemini call that returns TechniqueNotes. "
        "See the module docstring, and solutions/ if you get stuck."
    )
