"""Variant A — Debrief Coach. Reference solution.

To use it instead of your own, from the repository root::

    cp solutions/variants/coach.py lab/variants/coach.py

Two things are worth reading even if yours works.

**The Tracker is an ``AgentTool``, not a ``sub_agent``.** Sub-agent transfer
hands the conversation over: the Coach stops being the thing the trainee is
talking to, and the Tracker — which has no conversational instruction — answers
instead. Transfer hands over; AgentTool borrows. Most ADK material shows
transfer first, and it is the wrong shape here.

**Then ask whether the Tracker earned its place.** It could have been one
agent with a ``next_unreviewed`` tool, and that would be simpler. The
decomposition buys separation of concerns and somewhere to put procedural
state; at three tools that is not obviously worth it, and at twenty it would
be. Knowing which side of that line you are on is the take-home.
"""

from __future__ import annotations

from lab.lab3_agent import MODEL, TOOLS

TRACKER_INSTRUCTION = """\
You hold procedural state for one review session. You never speak to the
trainee; the coach calls you and relays what you say.

Answer only these questions, from the deviations list and what you have been
told was already discussed:
- which task steps this session contains, and which have flagged moments
- which flagged moments have not been reviewed yet
- which unreviewed moment is most worth taking next, and why in one clause

Prefer the largest measured gap when choosing. Be terse: this is machine
input, not conversation.
"""

COACH_INSTRUCTION = """\
You are debriefing a surgical trainee on one of their own recorded
training-exercise sessions. You are a coach reviewing practice footage, not a
clinician.

How to answer:
- Never state a number you did not get from a tool. If you need a figure, call
  one.
- Cite the step and timestamp a claim rests on, so the trainee can check it.
- Ask workflow_tracker what to look at next when the trainee wants direction
  rather than a specific answer. Relay its answer in your own words; never
  mention it by name.
- Call analyze_clip when they want to know what the footage shows, using the
  watch window from list_deviations.
- Pass on what the footage could not establish rather than dropping it. That
  is the honest part of the answer.
- Say plainly when the tools do not support an answer.
- These are efficiency observations about a training exercise. No diagnosis, no
  patients, no outcomes.
"""


def build_workflow_tracker():
    """Build the agent that holds procedural state for a session.

    Returns:
        A configured ``google.adk.Agent``. It never addresses the trainee.
    """
    from google.adk import Agent

    return Agent(
        name="workflow_tracker",
        model=MODEL,
        description=(
            "Knows which steps a session contains, which flagged moments have "
            "been reviewed, and which to take next. Ask it for direction, not "
            "for measurements."
        ),
        instruction=TRACKER_INSTRUCTION,
        tools=TOOLS,
    )


def build_coach(tracker=None):
    """Build the conversational Coach.

    Args:
        tracker: the WorkflowTracker. Omit it for the minimum shippable
            single-agent version, which is a perfectly honest answer.

    Returns:
        A configured ``google.adk.Agent``.
    """
    from google.adk import Agent
    from google.adk.tools import AgentTool

    tools = list(TOOLS)
    if tracker is not None:
        # AgentTool, not sub_agents: the Coach must keep the conversation.
        tools.append(AgentTool(agent=tracker))

    return Agent(
        name="debrief_coach",
        model=MODEL,
        description="Talks a trainee through their own recorded session.",
        instruction=COACH_INSTRUCTION,
        tools=tools,
    )


def build() -> object:
    """The full Coach, Tracker included. What the UI calls."""
    return build_coach(build_workflow_tracker())
