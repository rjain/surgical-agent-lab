"""Run an ADK agent and capture what it actually did.

SUPPLIED — you do not need to change this.

Building the agent is Lab 3's work. Driving it is not, so this module holds
the plumbing: one session, one turn, and a record of every tool call the model
made on the way to its answer.

That record is the point. An agent's prose is equally fluent whether it read a
measurement or invented one, and the only way to tell from the outside is to
look at what it called::

    reply = ask(agent, "Which step ran longest?")
    print(reply.text)
    for call in reply.calls:
        print(call.name, call.args, "->", call.response)

A turn with no calls behind a numeric claim is a turn that made the number up.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

APP_NAME = "surgical_agent_lab"
USER_ID = "participant"


@dataclass
class ToolCall:
    """One tool invocation, with whatever came back."""

    name: str
    args: dict
    response: object = None


@dataclass
class Reply:
    """One agent turn: the answer, and the evidence behind it."""

    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """Whether this turn consulted anything at all."""
        return bool(self.calls)


class Conversation:
    """A live ADK session against one agent.

    Holds the runner and session id so successive turns share history — the
    Coach needs that, or the trainee has to repeat themselves every question.
    """

    def __init__(self, agent) -> None:
        from google.adk.runners import InMemoryRunner

        self._runner = InMemoryRunner(agent, app_name=APP_NAME)
        self._loop = asyncio.new_event_loop()
        self._session = self._loop.run_until_complete(
            self._runner.session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID
            )
        )

    def ask(self, message: str) -> Reply:
        """Send one message and return the answer with its tool-call trace.

        Args:
            message: what the user said.

        Returns:
            A :class:`Reply`. ``reply.calls`` is empty when the model answered
            without consulting a tool, which is worth noticing.
        """
        from google.genai import types

        reply = Reply()
        pending: dict[str, ToolCall] = {}

        events = self._runner.run(
            user_id=USER_ID,
            session_id=self._session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        )
        for event in events:
            for part in (event.content.parts if event.content else []) or []:
                if getattr(part, "function_call", None):
                    call = ToolCall(
                        name=part.function_call.name,
                        args=dict(part.function_call.args or {}),
                    )
                    reply.calls.append(call)
                    # Keyed by name: responses arrive in later events, and the
                    # id field is not always populated.
                    pending[call.name] = call
                elif getattr(part, "function_response", None):
                    call = pending.get(part.function_response.name)
                    if call is not None:
                        call.response = part.function_response.response
                elif getattr(part, "text", None) and event.is_final_response():
                    reply.text += part.text
        return reply

    def close(self) -> None:
        """Release the event loop. Safe to call twice."""
        if not self._loop.is_closed():
            self._loop.close()


def ask(agent, message: str) -> Reply:
    """One-shot question against a fresh session.

    Convenient for a script or a test; use :class:`Conversation` when the
    turns need to remember each other.
    """
    talk = Conversation(agent)
    try:
        return talk.ask(message)
    finally:
        talk.close()
