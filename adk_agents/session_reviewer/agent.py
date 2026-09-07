"""Expose the lab's agent to ADK's dev UI.

Run from the repository root::

    adk web adk_agents

then open the printed URL and pick ``session_reviewer``. The Events pane
shows every tool call the agent makes — the same trace the Coach tab renders,
straight from Google's own inspector.

This loads the agent from your ``lab/lab3_agent.py``, so it only works once
you have implemented ``build_agent`` there (Lab 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

# `adk web adk_agents` puts adk_agents/ on sys.path, not the repository root,
# so `import lab` needs this.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab import config  # noqa: E402

config.load_env()  # the ADK server reads the key from the environment

try:
    from lab.lab3_agent import build_agent

    root_agent = build_agent()
except NotImplementedError:
    raise NotImplementedError(
        "adk web needs the agent from Lab 3: implement build_agent() in "
        "lab/lab3_agent.py, then reload this page."
    ) from None
