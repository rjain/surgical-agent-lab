"""Lab 3, shared step — wrap the capabilities as tools and build one agent.

YOU WRITE THIS FILE. Everyone does this before the variants diverge.

The three tools below are the entire interface both variants are written
against. Two already exist and only need wrapping; the third is your own Lab 2
work, which is the point — the thing you wrote becomes something the agent
decides to call on its own.

**The docstring is the contract.** ADK builds the function declaration the
model sees from the signature and the docstring, so a vague docstring produces
an agent that picks the wrong tool. Start by fixing the deliberately terrible
one on `get_metrics` in `lab/metrics.py` and watch the behaviour change.

**One standing instruction:** the agent must never state a number it did not
obtain from a tool. That is checkable, and the Coach tab shows the trace of
which tools were called so you can check it.
"""

from __future__ import annotations

from lab import env
from lab.cp1_analyze import analyze_clip  # noqa: F401  your Lab 2 work
from lab.metrics import get_metrics  # noqa: F401  supplied
from lab.rules import list_deviations  # noqa: F401  supplied

# The lab authenticates with a Gemini API key — no cloud project, no ADC, and
# no endpoint region to get wrong. `lab.env` reads the key from .env and points
# ADK at it, so there is nothing to configure twice.
MODEL = env.model()

#: The three capabilities every variant is built on.
TOOLS = [get_metrics, list_deviations, analyze_clip]


def build_agent():
    """Build one agent that answers open questions about a session.

    It should choose among the three tools rather than being told which to
    call, and ground every number in what a tool returned.

    Returns:
        A configured ``google.adk.Agent``.
    """
    raise NotImplementedError(
        "Lab 3: build an Agent with the three tools and an instruction that "
        "forbids unsourced numbers."
    )
