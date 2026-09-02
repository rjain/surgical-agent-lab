"""Lab 2 — reference solution.

One way to do it, not the only way. Announced at the start of the session as a
normal escape hatch: reading this after a genuine attempt is a good use of the
time, and being stuck for twenty minutes is not.

To use it instead of your own, from the repository root::

    cp solutions/lab2_analyze.py lab/lab2_analyze.py

Three things in here are worth reading even if your own version works:

* **The prompt is handed the flag**, not asked an open question. Compare the
  output of "what is wrong here" against "the rules found tool-swap churn at
  137s, describe what is visible" — the difference is large and it is the whole
  reason Layer 1 runs first.
* **The window comes from `Deviation.watch_window`**, not from the flag's full
  span. Flagged segments run 4 to 45 minutes and cost ~170,000 tokens whole;
  the 40-second watch window costs ~10,000. Sixteen times less, and it is what
  keeps twenty-five people inside a shared rate limit.
* **`validate()` fails loudly and the caller retries once** with a tightened
  instruction. A guardrail that only logs is not a guardrail.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from lab import clips, env
from lab.rules import find_deviations

# --- schema ----------------------------------------------------------------
# No tuples anywhere: tuple[float, float] serialises to `prefixItems`, which
# the Gemini API rejects outright.


class Observation(BaseModel):
    """One thing visible in the footage, tied to a moment."""

    t: float = Field(description="seconds from the start of the video part")
    what: str = Field(description="what is visibly happening")
    technique_note: str = Field(description="the coaching observation")
    confidence: Literal["clear", "probable", "uncertain"]


class TechniqueNotes(BaseModel):
    """Structured commentary on one window of a session."""

    case_id: str
    window_start_s: float
    window_end_s: float
    summary: str = Field(description="one or two sentences")
    observations: list[Observation] = Field(description="two to five, timestamped")
    visible_factors: list[str] = Field(
        description="what in frame might explain the flag"
    )
    not_visible: list[str] = Field(
        description="what this footage cannot establish. Never empty."
    )


class GuardrailViolation(ValueError):
    """The model returned something the lab will not pass on to a person."""


SYSTEM_INSTRUCTION = """\
You review recorded da Vinci TRAINING EXERCISE footage and comment on operative
technique for coaching purposes.

This is not a medical device and not clinical guidance. Do not diagnose, do not
refer to patients, and do not speculate about outcomes or complications.

Rules for every response:
- Cite a timestamp for every observation, and only timestamps inside the window
  you were given.
- Describe what is visible. Do not restate the measurement you were told.
- Populate not_visible with what this footage genuinely cannot establish. Never
  leave it empty.
"""

# Words that put the output outside the scope this lab is allowed to occupy.
CLINICAL_TERMS = (
    "diagnos",
    "patient",
    "complication",
    "injury",
    "injuries",
    "bleed",
    "adverse",
    "morbidity",
    "treatment",
)


# --- the guardrail ---------------------------------------------------------


def validate(notes: TechniqueNotes, t_start: float, t_end: float) -> None:
    """Refuse output that cannot be defended.

    Args:
        notes: what the model returned.
        t_start: the window start that was requested, in seconds.
        t_end: the window end that was requested, in seconds.

    Raises:
        GuardrailViolation: naming the specific check that failed.
    """
    if not notes.observations:
        raise GuardrailViolation("no observations returned")

    # A little tolerance: the model rounds, and clips carry some padding.
    slack = 2.0
    for obs in notes.observations:
        if not (t_start - slack) <= obs.t <= (t_end + slack):
            raise GuardrailViolation(
                f"observation at {obs.t:.1f}s is outside the requested window "
                f"{t_start:.1f}–{t_end:.1f}s — it was not shown that moment"
            )

    if not notes.not_visible:
        raise GuardrailViolation(
            "not_visible is empty — the model did not state what it could not see"
        )

    haystack = " ".join(
        [notes.summary, *notes.visible_factors, *notes.not_visible]
        + [f"{o.what} {o.technique_note}" for o in notes.observations]
    ).lower()
    for term in CLINICAL_TERMS:
        if term in haystack:
            raise GuardrailViolation(
                f"clinical vocabulary in the response ({term!r}) — this reviews "
                "a training exercise, not care"
            )


# --- the call --------------------------------------------------------------


def _prompt(case_id: str, part: int, t_start: float, t_end: float) -> str:
    """Hand the model the flag, so it describes rather than guesses."""
    context = ""
    for dev in find_deviations(case_id):
        if dev.part == part and dev.start_s < t_end and dev.end_s > t_start:
            context += (
                f"\n- {dev.rule_id} during {dev.step}, "
                f"{dev.start_s:.0f}–{dev.end_s:.0f}s. Measured: {dev.evidence}"
            )
    if not context:
        context = "\n- (no rule fired here; describe the technique on its own terms)"

    return (
        f"Session {case_id}, part {part}, window {t_start:.0f}–{t_end:.0f}s.\n"
        f"The deterministic rules engine flagged:{context}\n\n"
        "Describe what is visible in this window and what it suggests about "
        "technique. Cite timestamps. State what the footage cannot establish."
    )


def analyze_clip(
    case_id: str,
    part: int = 1,
    t_start: float | None = None,
    t_end: float | None = None,
) -> TechniqueNotes:
    """Describe what is visible in one window of a recorded session.

    Args:
        case_id: the session identifier, e.g. ``"case_045"``.
        part: the video part the window belongs to.
        t_start: window start in seconds within that part. Omit for the whole clip.
        t_end: window end in seconds within that part.
    """
    from google.genai import types

    clip = clips.find_for_window(
        case_id, part, t_start or 0.0, t_end or float("inf")
    )
    if clip is None:
        raise clips.ClipUnavailable(
            f"no clip in the manifest covers {case_id} part {part} "
            f"{t_start}–{t_end}s"
        )
    uri = clips.resolve_clip(clip.clip_id)

    # Offsets are relative to the clip, and the clip may start partway through
    # the part it was cut from.
    window = None
    if t_start is not None and t_end is not None:
        window = types.VideoMetadata(
            start_offset=f"{max(0.0, t_start - clip.start_s):.0f}s",
            end_offset=f"{max(1.0, t_end - clip.start_s):.0f}s",
        )
    lo = t_start if t_start is not None else clip.start_s
    hi = t_end if t_end is not None else clip.end_s

    prompt = _prompt(case_id, part, lo, hi)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=TechniqueNotes,
    )

    last_error: Exception | None = None
    for attempt in (1, 2):
        text = prompt if attempt == 1 else (
            prompt
            + "\n\nYour previous answer was rejected: "
            + str(last_error)
            + "\nAnswer again, fixing exactly that."
        )
        reply = env.client().models.generate_content(
            model=env.model(),
            contents=types.Content(
                role="user",
                parts=[
                    types.Part(
                        file_data=types.FileData(file_uri=uri, mime_type="video/mp4"),
                        video_metadata=window,
                    ),
                    types.Part(text=text),
                ],
            ),
            config=config,
        )
        notes = TechniqueNotes.model_validate_json(reply.text)
        try:
            validate(notes, lo, hi)
            return notes
        except GuardrailViolation as exc:
            last_error = exc

    raise GuardrailViolation(
        f"the model failed the guardrail twice; last reason: {last_error}"
    )
