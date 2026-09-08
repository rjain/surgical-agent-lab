"""Variant B — Session Auditor. Reference solution.

To use it instead of your own, from the repository root::

    cp solutions/variants/auditor.py lab/variants/auditor.py

The shape worth noticing: **the code decides what to fetch, and the model only
reasons over what it is handed.** ``gather`` is a plain function — it calls the
metrics, the rules and the clip analyser directly — and the agents downstream
get its output as their input. Compare with the Coach, where the model chooses
which tool to call and when.

Neither is more correct. The Coach is flexible and its behaviour varies run to
run; the Auditor is predictable and does exactly the same work every time. An
unattended report wants the second. A conversation wants the first. That
trade-off is the fork's real lesson, and you can only feel it having built one
and watched the other.

Two practical notes:

* ``max_concurrency`` is deliberately low. Fanning across every flagged moment
  at once is the single easiest way to exhaust a shared rate limit, and this
  agent is the one that does it. Four at a time is plenty.
* The small-sample caveat in ``limitations`` is hard-coded, not generated. Six
  curated sessions do not make a cohort, and a model asked to describe its own
  limitations will not reliably say so.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lab import config
from lab.lab2_analyze import analyze_clip
from lab.lab3_agent import MODEL
from lab.metrics import get_metrics
from lab.rules import list_deviations

#: How many clip analyses to run at once. See the module docstring.
MAX_CONCURRENCY = 4

SYNTHESISER_INSTRUCTION = """\
You compose one after-action report on a recorded surgical training exercise,
from findings you are given. You are reviewing practice footage, not care.

Rules:
- Use only the numbers in the input. Never introduce one of your own.
- Rank findings by the size of the measured gap, largest first.
- Every finding's evidence field must be the measurement it came from.
- Recommendations must follow from the findings, and name the step they apply to.
- Do not discuss diagnosis, patients or outcomes.
"""


class Finding(BaseModel):
    """One ranked observation about a session."""

    rank: int
    step: str
    t: float
    rule_id: str
    headline: str
    detail: str
    evidence: str = Field(description="the measurement that tripped the rule")


class SessionReport(BaseModel):
    """The after-action report for one session."""

    case_id: str
    headline: str
    duration_vs_cohort: str
    findings: list[Finding] = Field(description="ranked, most notable first")
    recommendations: list[str]
    limitations: list[str] = Field(description="what this review could not establish")


def build_segment_analyst():
    """The thing that describes one flagged window.

    This is Lab 2's ``analyze_clip`` — deliberately. The skeleton describes
    the analyst as an agent, and building it as one is a legitimate answer,
    but Lab 2 already produced a guardrailed, cached, schema-typed clip
    reader. Wrapping it in a second agent would add a model call, a second
    place for the guardrail to live, and nothing else.

    Reuse over re-architecture is the call to notice here.

    Returns:
        A callable ``(case_id, part, t_start, t_end) -> TechniqueNotes``.
    """
    return analyze_clip


def gather(case_id: str) -> dict:
    """Collect everything the report needs. No model involved.

    This is the half that makes the Auditor predictable: the same session
    always produces the same input, because a function decided what to fetch
    rather than an agent.

    Args:
        case_id: the session identifier, e.g. ``"case_045"``.
    """
    import concurrent.futures as futures

    summary = get_metrics(case_id)
    flags = list_deviations(case_id)

    def describe(flag: dict) -> dict:
        """One clip analysis, tolerant of a single failure."""
        try:
            notes = analyze_clip(
                case_id,
                int(flag["part"]),
                float(flag["watch_start_s"]),
                float(flag["watch_end_s"]),
            )
            return {"flag": flag, "notes": notes.model_dump()}
        except Exception as exc:
            # One unreadable clip must not lose the whole report.
            return {"flag": flag, "error": f"{type(exc).__name__}: {exc}"}

    with futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        segments = list(pool.map(describe, flags))

    return {"case_id": case_id, "summary": summary, "segments": segments}


def build_synthesizer():
    """Build the agent that turns gathered findings into one report.

    Returns:
        A configured ``google.adk.Agent`` with ``output_schema=SessionReport``.
    """
    from google.adk import Agent

    return Agent(
        name="synthesizer",
        model=MODEL,
        description="Composes one after-action report from gathered findings.",
        instruction=SYNTHESISER_INSTRUCTION,
        output_schema=SessionReport,
    )


def audit(case_id: str) -> SessionReport:
    """Review a whole session unattended and return the report.

    Args:
        case_id: the session identifier, e.g. ``"case_045"``.
    """
    import json

    from google.genai import types

    gathered = gather(case_id)
    reply = config.client().models.generate_content(
        model=MODEL,
        contents=(
            "Compose the report for this session.\n\n"
            + json.dumps(gathered, default=str)[:60000]
        ),
        config=types.GenerateContentConfig(
            system_instruction=SYNTHESISER_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=SessionReport,
            # The synthesiser reasons over the gathered JSON and calls
            # nothing. The SDK enables automatic function calling by
            # default, so say no explicitly: that default is what makes
            # the SDK log an AFC warning on a call that has no tools.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )
    report = SessionReport.model_validate_json(reply.text)

    # Hard-coded, not generated: six curated sessions are not a cohort, and a
    # model asked about its own limits will not reliably say so.
    caveat = (
        "Cohort medians derive from the curated subset, not a full cohort — "
        "treat comparisons as indicative."
    )
    if caveat not in report.limitations:
        report.limitations.append(caveat)
    return report


def build_auditor():
    """The audit pipeline the UI calls.

    Returns:
        A callable taking a case id and returning a :class:`SessionReport`.
    """
    return audit
