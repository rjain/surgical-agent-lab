"""Settings, read once, the same way from every entry point.

SUPPLIED — you do not need to change this file.

Everything in the lab authenticates with a **Gemini API key**. There is no
gcloud, no Application Default Credentials and no cloud project: one key in
``.env`` and you are working.

Editors disagree about ``.env`` files — VS Code and Antigravity load them for
Run and Debug but not for their integrated terminal, and Streamlit never reads
one — so this module reads the file itself. Terminal, debugger, Streamlit and
pytest then behave identically. Real environment variables always win over the
file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The key participants paste into `.env`.
KEY_VAR = "LAB_GEMINI_API_KEY"

#: Default model. `-latest` aliases exist on the Gemini API and track the
#: current release, so this does not go stale between runs of the lab.
DEFAULT_MODEL = "gemini-flash-latest"


def load_env() -> None:
    """Read ``.env`` from the repository root into the environment.

    Handles the two shapes people actually write by hand::

        LAB_GEMINI_API_KEY=abc123
        export LAB_GEMINI_API_KEY="abc123"

    Also points the Agent Development Kit at the same key, so nothing has to be
    configured twice.
    """
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value

    # The SDK and ADK read GOOGLE_API_KEY automatically, which is exactly why
    # the lab key has its own name: a participant who already works with Gemini
    # has GOOGLE_API_KEY set to their own key, and it would silently win.
    #
    # Assigned, not setdefault. Deferring to whatever is already in the
    # environment is what causes that shadowing — the lab key must win inside
    # this process. Nothing outside it is touched: this is os.environ for one
    # interpreter, not the participant's shell or their other projects.
    api_key = os.environ.get(KEY_VAR)
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    # GOOGLE_GENAI_USE_ENTERPRISE, not GOOGLE_GENAI_USE_VERTEXAI — the latter
    # is what every tutorial still shows, and both google-genai and ADK now
    # warn that it is deprecated. It followed the Vertex AI → Gemini Enterprise
    # rename. Only the new name is set: both libraries check it first, and
    # google-genai warns again if it finds the two disagreeing. An older SDK
    # that has never heard of it defaults to the Gemini API anyway, which is
    # what this line is asking for.
    os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "False"
    os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)


@lru_cache(maxsize=1)
def api_key() -> str:
    """The Gemini API key, or an empty string if it is not set."""
    load_env()
    return os.environ.get(KEY_VAR, "").strip()


@lru_cache(maxsize=1)
def model() -> str:
    """The model id the lab uses. Override with ``LAB_MODEL`` in ``.env``."""
    load_env()
    return os.environ.get("LAB_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


@lru_cache(maxsize=1)
def client():
    """A configured ``google.genai`` client, created once per process.

    Cached deliberately. Building a new client per call leaves the previous one
    to be garbage-collected, which closes the HTTP transport underneath it —
    the second call then fails with "Cannot send a request, as the client has
    been closed."

    Raises:
        RuntimeError: if no API key is set, with the fix in the message.
    """
    key = api_key()
    if not key:
        raise RuntimeError(
            f"No {KEY_VAR} found. Put it in .env at the repository root:\n"
            f'    {KEY_VAR}="your-key"\n'
            "Get a key from https://aistudio.google.com/apikey"
        )
    from google import genai

    return genai.Client(api_key=key)


def explain_api_error(exc: Exception) -> str:
    """Turn a Gemini API exception into something worth acting on.

    Args:
        exc: whatever the SDK raised.
    """
    text = str(exc)
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        if "credit" in text.lower() or "billing" in text.lower():
            return (
                "the key has no credit left — top it up at "
                "https://aistudio.google.com/ (Billing), then re-run"
            )
        return "rate limited — wait a moment and re-run, or use a key with a higher limit"
    if (
        "API_KEY_INVALID" in text
        or "API key not valid" in text
        or "UNAUTHENTICATED" in text
        or "401" in text
    ):
        return (
            f"the {KEY_VAR} in .env was rejected — check for a truncated paste, "
            "stray quotes, or a key that has been revoked. "
            "Get a fresh one from https://aistudio.google.com/apikey"
        )
    if "PERMISSION_DENIED" in text or "403" in text:
        return "the key is valid but not permitted to use this model"
    if "404" in text or "NOT_FOUND" in text:
        return "that model id does not exist on the Gemini API — check LAB_MODEL in .env"
    return "see the error above"
