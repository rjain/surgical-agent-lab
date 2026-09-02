"""Lab 3 shared step — reference solution.

To use it instead of your own, from the repository root::

    cp solutions/lab3_agent.py lab/lab3_agent.py

The interesting part is not `build_agent`, which is short. It is the
`get_metrics` docstring, shown here written properly. Compare it against the
one in the skeleton::

    def get_metrics(case_id: str, step: str = "") -> dict:
        \"\"\"gets metrics

        Args:
            case_id: the id
            step: the step
        \"\"\"

Both wrap the same working function. Asked *"where did I lose the most time in
case_045?"*, this is what actually happened on one run:

    vague   -> list_deviations(case_045)      # wrong tool first, wasted turn
               get_metrics(case_045)
               get_metrics(case_045, "Suturing")
               ...answered with a raw float: "ratio of 1.12552039966694"

    good    -> get_metrics(case_045)          # straight to the right tool
               get_metrics(case_045, "Suturing")
               list_deviations(case_045)
               ...answered "most time relative to the cohort median during Suturing"

Nothing about the body changed. ADK builds the declaration the model sees from
the signature and the docstring, and never reads the body at all.

Do run both yourself rather than taking the above on trust. Model behaviour
varies between runs and the difference shows up as *which tool it reaches for
first* and *how well it phrases the answer*, not as an outright failure. That
is worth knowing in itself: a bad tool description degrades an agent quietly
rather than breaking it.
"""

from __future__ import annotations

from lab import env
from lab.lab2_analyze import analyze_clip as _analyze_clip
from lab.metrics import get_metrics as _get_metrics
from lab.rules import list_deviations as _list_deviations

MODEL = env.model()

INSTRUCTION = """\
You help a surgical trainee review one of their own recorded training-exercise
sessions. You are a coach looking at practice footage, not a clinician.

How to answer:
- Never state a number you did not get from a tool. No estimates, no rounding
  from memory, no "roughly". If you need a figure, call a tool.
- Cite the step and the timestamp a claim comes from, so it can be checked.
- Call list_deviations first when asked what went wrong; it tells you where to
  look. Call get_metrics for timings. Call analyze_clip only when someone wants
  to know what the footage shows, and use the watch window it gives you.
- When the tools do not support an answer, say so plainly rather than filling
  the gap.
- These are efficiency observations about a training exercise. Do not discuss
  diagnosis, patients or outcomes.
"""


# --- the three tools -------------------------------------------------------


def get_metrics(case_id: str, step: str = "") -> dict:
    """Timing and instrument-usage measurements for a session, or one step of it.

    Use this for any question about how long something took, how a step
    compares with the corpus, or how much instrument swapping happened. Every
    number returned is measured from the session's own labels.

    Args:
        case_id: the session identifier, e.g. ``"case_045"``.
        step: optional task step name, e.g. ``"Suturing"``. Leave empty to get
            a summary of the whole session, including which step ran longest
            against its corpus median.
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
    ``watch_end_s`` from :func:`list_deviations`.

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

    Returns:
        A configured ``google.adk.Agent``.
    """
    from google.adk import Agent

    return Agent(
        name="session_reviewer",
        model=MODEL,
        description=(
            "Answers questions about one recorded training-exercise session, "
            "grounded in its measurements."
        ),
        instruction=INSTRUCTION,
        tools=TOOLS,
    )
