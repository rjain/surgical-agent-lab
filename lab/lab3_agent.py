"""Lab 3, shared step — wrap the capabilities as tools and build one agent.

YOU WRITE THIS FILE. Everyone does this before the variants diverge.

Three tools are the entire interface both variants are written against. Two
already exist and only need wrapping; the third is your own Lab 2 work, which
is the point — the thing you wrote becomes something the agent decides to call
on its own.

**The docstring is the contract.** ADK builds the function declaration the
model sees from the signature and the docstring. It never reads the body. So a
vague docstring produces an agent that picks the wrong tool, or calls the right
one with nonsense arguments, and no amount of fixing the body will help.

Start by reading the three wrappers below. One of them has a docstring written
the way people write them when they are in a hurry. Run the agent, watch what
it does with the question *"where did I lose the most time in case_045?"*, then
rewrite that docstring properly and run it again. That comparison is the whole
lesson of this checkpoint, and it takes about five minutes.

**One standing instruction for the agent:** never state a number it did not
obtain from a tool. It is checkable, and the Coach tab shows the trace of which
tools were called so you can check it.
"""

from __future__ import annotations

from lab import config
from lab.lab2_analyze import analyze_clip as _analyze_clip
from lab.metrics import get_metrics as _get_metrics
from lab.rules import list_deviations as _list_deviations

MODEL = config.model()


# --- the three tools -------------------------------------------------------
# These are thin wrappers. The bodies are correct; the docstrings are what the
# model sees, and one of them is not good enough.


def get_metrics(case_id: str, step: str = "") -> dict:
    """gets metrics

    Args:
        case_id: the id
        step: the step
    """
    return _get_metrics(case_id, step or None)


def list_deviations(case_id: str) -> list[dict]:
    """Flagged moments in a recorded session, worth a reviewer's attention.

    Each entry gives the task step, which rule fired, the measurement behind
    it, and the stretch of footage worth watching. Use ``watch_start_s`` and
    ``watch_end_s`` when you want to look at what happened.

    Args:
        case_id: the session identifier, e.g. ``"case_045"``.
    """
    return _list_deviations(case_id)


def analyze_clip(
    case_id: str, part: int = 1, t_start: float = 0.0, t_end: float = 0.0
) -> dict:
    """Describe what is visible in one window of a recorded session's footage.

    Use this to explain a flagged moment — what the trainee actually did, as
    opposed to the measurement that flagged it. Pass the ``watch_start_s`` and
    ``watch_end_s`` from :func:`list_deviations`; those windows are sized to be
    affordable, and the full step span is far too long to analyse.

    Args:
        case_id: the session identifier, e.g. ``"case_045"``.
        part: the video part the window belongs to; time restarts in each part.
        t_start: window start in seconds within that part.
        t_end: window end in seconds within that part.
    """
    notes = _analyze_clip(case_id, part, t_start or None, t_end or None)
    return notes.model_dump()


#: Every variant is built on exactly these.
TOOLS = [get_metrics, list_deviations, analyze_clip]


def build_agent():
    """Build one agent that answers open questions about a session.

    It should choose among the three tools rather than being told which to
    call, and ground every number in what a tool returned.

    Returns:
        A configured ``google.adk.Agent``.
    """
    raise NotImplementedError(
        "Lab 3: build an Agent with TOOLS and an instruction that forbids "
        "unsourced numbers. See solutions/ if you get stuck."
    )
