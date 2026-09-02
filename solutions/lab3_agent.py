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
case_045?"* four times each, the difference is in how the agent gets there:

    run   variant   tool calls   first tool called
      1   vague              5   list_deviations   <- wrong tool, wasted turn
      1   good               4   get_metrics
      2   vague              5   list_deviations   <- again
      2   good               3   get_metrics
      3   vague              3   get_metrics
      3   good               4   get_metrics
      4   vague             11   get_metrics       <- thrashing
      4   good               3   get_metrics

Nothing about the body changed. ADK builds the declaration the model sees from
the signature and the docstring, and never reads the body at all.

**Read that table carefully, because the honest lesson is narrower than it
first looks.** The good docstring is *consistent* — always the right tool
first, always three or four calls. The vague one is *erratic*: right half the
time, and on run 4 it made eleven calls to answer one question. Neither ever
outright failed. A bad tool description does not break an agent; it makes it
expensive and unpredictable, which is much harder to notice in review and much
worse in production.

What the docstring does **not** fix is how the answer is phrased. An earlier
version of this file claimed it did — that the vague agent quoted a raw
`1.12552039966694` and the good one read it back properly. Measured, both did
it, every time. That is an *instruction* problem, and it is fixed by the
rounding line in `INSTRUCTION` above, not by anything in a docstring. Two
different failure modes with two different remedies, and worth keeping
straight.

Run it yourself rather than taking the table on trust; the counts vary.
"""

from __future__ import annotations

from lab import config
from lab.lab2_analyze import analyze_clip as _analyze_clip
from lab.metrics import get_metrics as _get_metrics
from lab.rules import list_deviations as _list_deviations

MODEL = config.model()

INSTRUCTION = """\
You help a surgical trainee review one of their own recorded training-exercise
sessions. You are a coach looking at practice footage, not a clinician.

How to answer:
- Never state a number you did not get from a tool. No estimates, no rounding
  from memory, no "roughly". If you need a figure, call a tool.
- Report figures the way a person would read them aloud: durations in minutes
  to one decimal, ratios to two. The tools return full precision; quoting it
  back ("1.12552039966694x") is noise, not rigour.
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
