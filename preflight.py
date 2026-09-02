#!/usr/bin/env python3
"""Check this machine is ready for the lab, and say exactly how to fix it.

Run this once when you receive your credentials, well before the session::

    python preflight.py

Every check prints PASS, FAIL or SKIP with the fix on the line beneath. The
report at the bottom is what you send back to the instructors — copy the whole
thing, including the failures. A failure found today takes two minutes to fix;
the same failure on the day costs the room fifteen.

The one check that costs anything is skipped unless you pass ``--cloud``.
Run the full set at least once::

    python preflight.py --cloud
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import socket
import sys
from pathlib import Path

from lab import env

MIN_PYTHON = (3, 10)
STREAMLIT_PORT = 8501

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results: list[tuple[str, str, str, str]] = []


def record(status: str, name: str, detail: str = "", fix: str = "") -> None:
    _results.append((status, name, detail, fix))
    icon = {PASS: "  ok  ", FAIL: " FAIL ", SKIP: " skip "}[status]
    print(f"[{icon}] {name}" + (f" — {detail}" if detail else ""))
    if status == FAIL and fix:
        print(f"          fix: {fix}")


# --- 1-3: the interpreter and its packages ---------------------------------


def check_python() -> None:
    got = sys.version_info[:3]
    if got >= MIN_PYTHON:
        record(PASS, "Python version", ".".join(map(str, got)))
    else:
        record(
            FAIL,
            "Python version",
            f"{'.'.join(map(str, got))}, need >= {'.'.join(map(str, MIN_PYTHON))}",
            "install a newer Python, then recreate the virtual environment",
        )


def check_virtualenv() -> None:
    active = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if active:
        record(PASS, "Virtual environment", Path(sys.prefix).name)
    else:
        record(
            FAIL,
            "Virtual environment",
            "running against system Python",
            "python -m venv .venv && source .venv/bin/activate  "
            "(Windows: .venv\\Scripts\\activate)",
        )


def check_packages() -> None:
    wanted = {
        "pandas": "pandas",
        "streamlit": "streamlit",
        "google.adk": "google-adk",
        "google.genai": "google-genai",
        "pydantic": "pydantic",
    }
    missing = []
    for module, dist in wanted.items():
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(dist)
    if missing:
        record(
            FAIL,
            "Required packages",
            f"missing: {', '.join(missing)}",
            "pip install -r requirements.txt",
        )
    else:
        import streamlit

        record(PASS, "Required packages", f"streamlit {streamlit.__version__}")


# --- 4: the one credential ------------------------------------------------


def check_api_key() -> None:
    """The one credential the lab needs."""
    key = env.api_key()
    if not key:
        record(
            FAIL, "Gemini API key", f"{env.KEY_VAR} not set",
            "put it in .env at the repository root, then re-run. "
            "Get a key from https://aistudio.google.com/apikey",
        )
        return
    if len(key) < 20:
        record(
            FAIL, "Gemini API key", f"{env.KEY_VAR} looks too short ({len(key)} chars)",
            "check for stray quotes or a truncated paste in .env",
        )
        return
    record(PASS, "Gemini API key", f"set, {len(key)} chars, ending {key[-4:]}")


# --- 5-7: the things that actually fail on a locked-down machine -----------


def check_model_call(enabled: bool) -> None:
    """One real call. The only check that proves the key actually works.

    A key can be present, correctly formatted and still unusable — most often
    because its credit has run out, which returns a 429 that says nothing about
    this lab unless you read it carefully.
    """
    if not enabled:
        record(SKIP, "Model endpoint reachable", "pass --cloud to run this")
        return
    if not env.api_key():
        record(SKIP, "Model endpoint reachable", "no API key to test with")
        return
    model = env.model()
    try:
        client = env.client()
        reply = client.models.generate_content(model=model, contents="Reply with OK.")
        text = (getattr(reply, "text", "") or "").strip()[:20]
        record(PASS, "Model endpoint reachable", f"{model} replied {text!r}")
    except Exception as exc:
        record(
            FAIL,
            "Model endpoint reachable",
            f"{type(exc).__name__}: {str(exc)[:100]}",
            env.explain_api_error(exc),
        )


def check_dataset() -> None:
    from lab.data import DATA_DIR, list_cases

    cases = list_cases()
    if cases:
        record(PASS, "Dataset readable", f"{len(cases)} cases under {DATA_DIR}")
        return
    record(
        FAIL,
        "Dataset readable",
        f"no cases under {DATA_DIR}",
        "set LAB_DATA_DIR to the folder holding case_* directories",
    )


def check_pipeline_end_to_end() -> None:
    """Load a case, measure it, run the rules. Catches a broken install fast."""
    try:
        from lab.data import list_cases
        from lab.rules import find_deviations

        cases = list_cases()
        if not cases:
            record(SKIP, "Detection pipeline", "no dataset to run against")
            return
        target = "case_045" if "case_045" in cases else cases[0]
        found = find_deviations(target)
        record(PASS, "Detection pipeline", f"{target}: {len(found)} flags")
    except Exception as exc:
        record(
            FAIL,
            "Detection pipeline",
            f"{type(exc).__name__}: {str(exc)[:110]}",
            "reinstall dependencies, then re-run; if it persists, send this report",
        )


# --- 8: the local web server -----------------------------------------------


def check_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", STREAMLIT_PORT))
            record(PASS, f"Port {STREAMLIT_PORT} free", "Streamlit can start")
        except OSError:
            record(
                FAIL,
                f"Port {STREAMLIT_PORT} free",
                "already in use",
                f"stop whatever is on {STREAMLIT_PORT}, or run "
                f"streamlit run ui/app.py --server.port 8502",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="also make one real model call (costs a fraction of a cent)",
    )
    args = parser.parse_args()

    env.load_env()

    print(f"Surgical Agent Lab — preflight on {platform.platform()}\n")
    check_python()
    check_virtualenv()
    check_packages()
    check_api_key()
    check_model_call(args.cloud)
    check_dataset()
    check_pipeline_end_to_end()
    check_port()

    failed = [r for r in _results if r[0] == FAIL]
    skipped = [r for r in _results if r[0] == SKIP]
    print("\n" + "-" * 68)
    print("COPY EVERYTHING BELOW THIS LINE INTO YOUR REPLY")
    print("-" * 68)
    print(f"python   {sys.version.split()[0]}   platform {platform.platform()}")
    for status, name, detail, _ in _results:
        print(f"{status:5} {name:32} {detail}")
    print(
        f"\n{len(_results) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} skipped"
    )
    if not failed and not skipped:
        print("Ready. Nothing further to do before the session.")
    elif not failed:
        print("Ready, but re-run with --cloud to check the parts that need the network.")
    else:
        print("Not ready. Apply the fixes above and re-run.")
    print("-" * 68)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
