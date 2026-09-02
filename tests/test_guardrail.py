"""Tests for the Lab 2 guardrail — the reference `validate()`.

No key, no tokens, no network: `validate()` is a pure function over a model
that has already come back, which is exactly why it is testable and exactly
why it is where the lab puts the honesty.

These are here for two reasons. The guardrail is the teaching centrepiece of
Lab 2 and had no test at all, and a participant comparing their own
`validate()` against the reference needs to know what the reference actually
promises. Each test names the failure it prevents.

The subject is `solutions/`, deliberately — `lab/lab2_analyze.py` is a
skeleton that raises, and `tests/test_supplied.py` has a separate test that
keeps it that way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solutions.lab2_analyze import (  # noqa: E402
    CLINICAL_TERMS,
    GuardrailViolation,
    Observation,
    TechniqueNotes,
    validate,
)

WINDOW = (100.0, 140.0)


def notes(**overrides) -> TechniqueNotes:
    """A response that passes, so each test can break exactly one thing."""
    payload = {
        "case_id": "case_045",
        "window_start_s": WINDOW[0],
        "window_end_s": WINDOW[1],
        "summary": "The forceps holds position while dissection continues.",
        "observations": [
            Observation(
                t=110.0,
                what="the forceps maintains traction",
                technique_note="steady, no repositioning",
                confidence="clear",
            )
        ],
        "visible_factors": ["the instrument leaves frame briefly"],
        "not_visible": ["console pedal input"],
    }
    payload.update(overrides)
    return TechniqueNotes(**payload)


def test_a_defensible_response_passes():
    validate(notes(), *WINDOW)


def test_an_empty_response_is_refused():
    """A model that returns nothing must not read as a clean session."""
    with pytest.raises(GuardrailViolation, match="no observations"):
        validate(notes(observations=[]), *WINDOW)


# Slack is 2s, so 98.0-142.0 is accepted; these sit outside it.
@pytest.mark.parametrize("t", [40.0, 97.9, 142.1, 900.0])
def test_a_timestamp_outside_the_window_is_refused(t):
    """The failure this exists for: a cited moment the model was never shown.

    The schema cannot catch it — 40.0 is a perfectly valid float — so nothing
    but this check stands between a hallucinated timestamp and a trainee.
    """
    bad = notes(
        observations=[
            Observation(t=t, what="x", technique_note="y", confidence="clear")
        ]
    )
    with pytest.raises(GuardrailViolation, match="outside the requested window"):
        validate(bad, *WINDOW)


@pytest.mark.parametrize("t", [98.0, 100.0, 140.0, 142.0])
def test_the_slack_tolerates_rounding_at_the_edges(t):
    """Two seconds either side. The model rounds and clips carry padding, so a
    boundary observation is honest, not invented."""
    ok = notes(
        observations=[
            Observation(t=t, what="x", technique_note="y", confidence="clear")
        ]
    )
    validate(ok, *WINDOW)


def test_an_empty_not_visible_is_refused():
    """Models overclaim on medical-adjacent video. Making honesty a required
    field is the fix, and this is what enforces it."""
    with pytest.raises(GuardrailViolation, match="not_visible is empty"):
        validate(notes(not_visible=[]), *WINDOW)


@pytest.mark.parametrize("term", CLINICAL_TERMS)
def test_every_clinical_term_is_caught_in_the_summary(term):
    """Parametrised over the real tuple, so adding a term without a test is
    not possible."""
    with pytest.raises(GuardrailViolation, match="clinical vocabulary"):
        validate(notes(summary=f"There was {term}ing visible."), *WINDOW)


@pytest.mark.parametrize(
    "field",
    ["visible_factors", "not_visible", "what", "technique_note"],
)
def test_clinical_vocabulary_is_caught_in_every_field_not_just_the_summary(field):
    """A term hiding in an observation reaches the reader just as easily."""
    if field in ("visible_factors", "not_visible"):
        bad = notes(**{field: ["risk of patient harm"]})
    else:
        bad = notes(
            observations=[
                Observation(
                    t=110.0,
                    what="bleeding at the margin" if field == "what" else "x",
                    technique_note="patient outcome" if field == "technique_note" else "y",
                    confidence="clear",
                )
            ]
        )
    with pytest.raises(GuardrailViolation, match="clinical vocabulary"):
        validate(bad, *WINDOW)


def test_the_check_is_case_insensitive():
    """A capitalised term is the same term."""
    with pytest.raises(GuardrailViolation, match="clinical vocabulary"):
        validate(notes(summary="PATIENT positioning was adjusted."), *WINDOW)


def test_the_violation_names_which_check_failed():
    """The message is fed back into the prompt on retry, so it has to be
    specific enough for the model to act on."""
    with pytest.raises(GuardrailViolation) as caught:
        validate(
            notes(
                observations=[
                    Observation(t=999.0, what="x", technique_note="y", confidence="clear")
                ]
            ),
            *WINDOW,
        )
    message = str(caught.value)
    assert "999.0" in message
    assert "100.0" in message and "140.0" in message
