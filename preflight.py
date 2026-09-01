#!/usr/bin/env python3
"""Check this machine is ready for the lab, and say exactly how to fix it.

Run this once when you receive your credentials, well before the session::

    python preflight.py

Every check prints PASS, FAIL or SKIP with the fix on the line beneath. The
report at the bottom is what you send back to the instructors — copy the whole
thing, including the failures. A failure found today takes two minutes to fix;
the same failure on the day costs the room fifteen.

Checks that spend money or need cloud access are skipped unless you pass
``--cloud``. Run the full set at least once::

    python preflight.py --cloud
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

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


# --- 4-6: cloud tooling and identity ---------------------------------------


def check_gcloud() -> None:
    if shutil.which("gcloud") is None:
        record(
            FAIL,
            "gcloud CLI",
            "not on PATH",
            "install the Google Cloud CLI: https://cloud.google.com/sdk/docs/install",
        )
        return
    try:
        out = subprocess.run(
            ["gcloud", "version"], capture_output=True, text=True, timeout=30
        )
        first = out.stdout.strip().splitlines()[0] if out.stdout else "installed"
        record(PASS, "gcloud CLI", first)
    except Exception as exc:
        record(FAIL, "gcloud CLI", str(exc), "reinstall the Google Cloud CLI")


def check_adc() -> None:
    explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit and Path(explicit).exists():
        record(PASS, "Application Default Credentials", "from environment")
        return
    default = (
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    )
    windows = (
        Path(os.environ.get("APPDATA", "")) / "gcloud"
        / "application_default_credentials.json"
    )
    if default.exists() or windows.exists():
        record(PASS, "Application Default Credentials", "present")
    else:
        record(
            FAIL,
            "Application Default Credentials",
            "not found",
            "gcloud auth application-default login",
        )


def _gcloud(*args: str) -> str:
    if not shutil.which("gcloud"):
        return ""
    try:
        out = subprocess.run(
            ["gcloud", *args], capture_output=True, text=True, timeout=30
        )
        value = out.stdout.strip()
        return "" if value in ("", "(unset)") else value
    except Exception:
        return ""


def active_project() -> str:
    """The project the SDKs will actually use."""
    return os.environ.get("GOOGLE_CLOUD_PROJECT") or _gcloud(
        "config", "get-value", "project"
    )


def check_project() -> None:
    """Confirm the project is set — and that it is the *right* one.

    Most developers already have gcloud configured for something else. Without
    the expected value to compare against, this check would go green while
    pointing at a personal project, and ``--cloud`` would bill that project
    instead of the training one.
    """
    project = active_project()
    account = _gcloud("config", "get-value", "account")
    expected = os.environ.get("LAB_PROJECT_ID", "").strip()
    who = f"{project} ({account})" if account else project

    if not project:
        record(
            FAIL,
            "Cloud project",
            "not set",
            "gcloud config set project <the project id in your welcome email>",
        )
        return

    if not expected:
        record(
            PASS,
            "Cloud project",
            f"{who} — not verified, LAB_PROJECT_ID unset",
        )
        return

    if project == expected:
        record(PASS, "Cloud project", who)
    else:
        record(
            FAIL,
            "Cloud project",
            f"using {who}, expected {expected}",
            f"gcloud config set project {expected}   "
            f"(and check `gcloud config configurations list` — you may be on "
            f"another configuration)",
        )


# --- 7-9: the things that actually fail on a corporate network -------------


def check_vertex(enabled: bool) -> None:
    if not enabled:
        record(SKIP, "Vertex AI reachable", "pass --cloud to run this")
        return
    project = active_project()
    # Gemini 3.x is served only from the `global` endpoint on Vertex; asking a
    # regional endpoint for it returns 404 NOT_FOUND, which reads like a
    # permissions problem and is not one.
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    expected = os.environ.get("LAB_PROJECT_ID", "").strip()
    if expected and project != expected:
        record(
            SKIP,
            "Vertex AI reachable",
            f"refusing to bill {project}; fix the project first",
        )
        return
    model = os.environ.get("LAB_MODEL", "gemini-3.5-flash")
    try:
        from google import genai

        client = genai.Client(vertexai=True, project=project, location=location)
        reply = client.models.generate_content(model=model, contents="Reply with OK.")
        text = (getattr(reply, "text", "") or "").strip()[:20]
        record(PASS, "Vertex AI reachable", f"{model} @ {location} replied {text!r}")
    except Exception as exc:
        detail = f"{type(exc).__name__}: {str(exc)[:100]}"
        if "404" in str(exc) or "NOT_FOUND" in str(exc):
            fix = (
                f"{model} is not served from {location!r}. Gemini 3.x is only on "
                "the 'global' endpoint — set GOOGLE_CLOUD_LOCATION=global, or use "
                "gemini-2.5-flash for a regional endpoint."
            )
        else:
            fix = (
                "check the project id, that the Vertex AI API is enabled, and "
                "that your network allows googleapis.com"
            )
        record(FAIL, "Vertex AI reachable", detail, fix)


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


# --- 10: the local web server ----------------------------------------------


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
        help="also make one real Vertex AI call (costs a fraction of a cent)",
    )
    args = parser.parse_args()

    # data.py reads .env, so LAB_PROJECT_ID from there is picked up here too.
    try:
        from lab.data import _load_dotenv

        _load_dotenv()
    except Exception:
        pass

    print(f"Surgical Agent Lab — preflight on {platform.platform()}\n")
    check_python()
    check_virtualenv()
    check_packages()
    check_gcloud()
    check_adc()
    check_project()
    check_vertex(args.cloud)
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
