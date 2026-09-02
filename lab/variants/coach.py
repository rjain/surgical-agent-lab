"""Variant A — Debrief Coach. An agent you talk to.

YOU WRITE THIS FILE if your group chose Variant A.

A trainee asks about a recorded session and gets an answer tied to a real
measurement and a real moment, never to the model's impression of the video.

**Minimum shippable:** one Coach agent holding the three tools. It works, it
demonstrates, and it is an honest answer. Reach this first.

**Fuller version:** add a WorkflowTracker that owns procedural knowledge —
which steps exist, which have been discussed, what is worth reviewing next.

**Wire the Tracker as an ``AgentTool``, not as a ``sub_agent``.** Sub-agent
transfer hands the conversation over, so the Coach disappears mid-sentence and
the trainee ends up talking to the tracker. Transfer hands over; AgentTool
borrows. Most ADK material shows transfer first — do not copy it here.

When it works, ask whether the Tracker earned its place. Could this have been
one agent with a `next_deviation()` tool? Usually yes. Knowing when *not* to
decompose is the take-home.
"""

from __future__ import annotations

from lab.lab3_agent import MODEL, TOOLS  # noqa: F401


def build_workflow_tracker():
    """Build the agent that holds procedural state for a session.

    It should know which steps exist, which have already been discussed, and
    what is worth reviewing next. It never addresses the trainee directly.

    Returns:
        A configured ``google.adk.Agent``.
    """
    raise NotImplementedError("Variant A: build the WorkflowTracker.")


def build_coach(tracker=None):
    """Build the conversational Coach.

    Args:
        tracker: the WorkflowTracker, wrapped as an ``AgentTool``. Omit it for
            the minimum shippable single-agent version.

    Returns:
        A configured ``google.adk.Agent`` holding the three tools.
    """
    raise NotImplementedError("Variant A: build the Coach.")
