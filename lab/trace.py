"""Report what the pipeline is doing, while it does it.

SUPPLIED — you do not need to change this, but you should call it.

Lab 2 looks like one function call and is really five things: find the clip,
check the instructors' upload is still alive, upload your own copy if it is
not, send the window to the model, then check what comes back. When all of
that hides behind a spinner, a slow run and a broken run look identical, and
"it is not working" is the only report anyone can give.

So the supplied parts announce each step, and the interface prints them as
they happen. Add your own from inside ``analyze_clip``::

    from lab import trace

    trace.step(f"calling {config.model()} with a {t_end - t_start:.0f}s window")

Nothing is listening outside the interface, so a call from a script or a test
costs one function call and produces no output.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

#: Callables to notify. Module-level on purpose: the interface adds one for
#: the duration of a click, and everything under that click reports through
#: it without a context object being threaded down five call frames.
_listeners: list[Callable[[str], None]] = []


def step(message: str) -> None:
    """Announce one step. Does nothing when nobody is listening.

    Args:
        message: what is happening, in the present tense, as a person would
            say it. "uploading case_045_p1_4434.mp4 (1.2 MB)" rather than
            "upload_clip: begin".
    """
    for listener in list(_listeners):
        try:
            listener(message)
        except Exception:
            # A broken listener must never take down the pipeline it is
            # only watching.
            pass


@contextmanager
def listening(callback: Callable[[str], None]) -> Iterator[None]:
    """Receive every :func:`step` raised inside this block.

    Args:
        callback: called with each message, in order.
    """
    _listeners.append(callback)
    try:
        yield
    finally:
        if callback in _listeners:
            _listeners.remove(callback)
