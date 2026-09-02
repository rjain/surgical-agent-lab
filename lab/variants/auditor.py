"""Variant B — Session Auditor. An agent you launch and leave.

YOU WRITE THIS FILE if your group chose Variant B.

One action reviews a whole session unattended and emits a typed report:
timeline, flagged moments, ranked findings, recommendations, and an explicit
statement of what the footage could not establish.

**Minimum shippable:** a plain Python loop over the session's flagged windows
into one analyst call, then a single synthesis call. No graph primitives
needed. Reach this first.

**Fuller version:** a ``Workflow`` graph::

    START -> fetch_clip_data (FunctionNode)
          -> segment_analyst (Agent, fanned across the windows)
          -> synthesizer     (Agent, output_schema=SessionReport)

Each node's return value is passed to the next as its input, so there is no
session state to plumb. Set ``max_concurrency`` to fan out.

Note what differs from Variant A: here the *code* decides what to fetch and
the model only reasons over what it is handed. That is the working trade-off
between flexibility and predictability, and both shapes turn up in production.

**On the limitations field:** hard-code the small-sample caveat rather than
letting the model generate it. With six curated sessions, "cohort median" is a
small-n statistic and the report must say so.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lab.lab3_agent import MODEL  # noqa: F401


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
    limitations: list[str] = Field(
        description="what this review could not establish"
    )


def build_segment_analyst():
    """Build the agent that describes one flagged window.

    Returns:
        A configured ``google.adk.Agent`` with an output schema.
    """
    raise NotImplementedError("Variant B: build the SegmentAnalyst.")


def build_synthesizer():
    """Build the agent that composes the findings into one report.

    Returns:
        A configured ``google.adk.Agent`` with ``output_schema=SessionReport``.
    """
    raise NotImplementedError("Variant B: build the Synthesizer.")


def build_auditor():
    """Build the whole audit pipeline.

    Returns:
        Either a ``google.adk.Workflow`` or a callable taking a case id and
        returning a :class:`SessionReport`.
    """
    raise NotImplementedError("Variant B: assemble the pipeline.")
