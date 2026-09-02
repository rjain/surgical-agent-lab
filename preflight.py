#!/usr/bin/env python3
"""Check this machine is ready for the lab, and say exactly how to fix it.

Run this once when you receive your credentials, well before the session::

    python preflight.py

Every check prints PASS, FAIL or SKIP with the fix on the line beneath. The
report at the bottom is what you send back to the instructors — copy the whole
thing, including the failures. A failure found today takes two minutes to fix;
the same failure on the day costs the room fifteen.

Ten checks: Python version, virtual environment, packages, API key, network
reach to the API, dataset, the detection pipeline end to end, the Streamlit
port, **the Antigravity extensions the Run and Debug targets need**, and
whether this key can see the lab's video clips. The last two are warnings
rather than failures — the IDE is what the session
assumes, but everything here also runs from a terminal.

**Nothing here spends a token.** The one check that touches the API lists the
available models, which is free — enough to prove your network reaches Google,
that TLS inspection has not broken the SDK, and that your key works.

Instructors funding the keys can add ``--spend`` to make one real billed call
and confirm a key has credit on it. Participants never need that.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import socket
import sys
from pathlib import Path

from lab import config

MIN_PYTHON = (3, 10)
STREAMLIT_PORT = 8501

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIP", "WARN"
_results: list[tuple[str, str, str, str]] = []


def record(status: str, name: str, detail: str = "", fix: str = "") -> None:
    _results.append((status, name, detail, fix))
    icon = {PASS: "  ok  ", FAIL: " FAIL ", SKIP: " skip ", WARN: " note "}[status]
    print(f"[{icon}] {name}" + (f" — {detail}" if detail else ""))
    if status in (FAIL, WARN) and fix:
        print(f"          {'fix' if status == FAIL else 'do this'}: {fix}")


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
    key = config.api_key()
    if not key:
        # Keys are handed out on the day, so an empty one before then is
        # expected rather than broken. Everything else in this report still
        # tells you whether the machine is ready.
        record(
            WARN, "Gemini API key", f"{config.KEY_VAR} not set yet",
            "expected if you have not been given a key. Paste it into .env on "
            "the day and re-run this — it takes thirty seconds",
        )
        return
    if len(key) < 20:
        record(
            FAIL, "Gemini API key", f"{config.KEY_VAR} looks too short ({len(key)} chars)",
            "check for stray quotes or a truncated paste in .env",
        )
        return
    # Four trailing characters, deliberately: this report gets copied into an
    # email, and with twenty-five individually-issued keys "which key are you
    # on?" is a real diagnostic question. Four of 53 characters gives away
    # nothing useful. Masking the resolved value is safe in a way that
    # pattern-matching .env is not — see AGENTS.md.
    record(PASS, "Gemini API key", f"set, {len(key)} chars, ending {key[-4:]}")


# --- 5-7: the things that actually fail on a locked-down machine -----------


def check_api_reachable() -> None:
    """Reach the API and confirm the model exists. Costs nothing.

    Listing models is a free operation, which is the whole point: this runs on
    every machine before the day without spending a token, and it still proves
    the three things that actually go wrong — the network reaches
    generativelanguage.googleapis.com, corporate TLS inspection has not broken
    the SDK's certificate chain, and the key authenticates.

    It does not prove the key has credit or quota. That is deliberate: funding
    is the instructors' problem, not something to discover from twenty-five
    machines.
    """
    if not config.api_key():
        record(
            SKIP, "API reachable", "no key yet — re-run once you have one"
        )
        return
    wanted = config.model()
    try:
        names = {
            (m.name or "").split("/")[-1] for m in config.client().models.list()
        }
    except Exception as exc:
        record(
            FAIL,
            "API reachable",
            f"{type(exc).__name__}: {str(exc)[:100]}",
            config.explain_api_error(exc),
        )
        return

    if wanted in names:
        record(PASS, "API reachable", f"{len(names)} models listed, {wanted} among them")
    else:
        close = sorted(n for n in names if "flash" in n)[:3]
        record(
            FAIL,
            "API reachable",
            f"reached the API, but {wanted} is not offered to this key",
            f"set LAB_MODEL in .env to one that is, e.g. {', '.join(close) or 'see aistudio.google.com'}",
        )


def check_clip_visibility() -> None:
    """Whether this key can see the clips the lab will ask it to read.

    Costs nothing: reading a file's metadata is not a billed operation, so
    this works on a key with no generation quota left.

    It is the one thing the other checks cannot tell you. Files on the Gemini
    API belong to the *project* the key came from, so a perfectly valid,
    fully funded key from the wrong project passes every check above and then
    fails on every clip in Lab 2.

    Never a failure, because two innocent situations look identical here: the
    instructors have not refreshed the clips yet (they expire after 48 hours),
    and you are on the shared pre-flight key, which lives in its own project
    by design. Both report the same 403.
    """
    if not config.api_key():
        record(SKIP, "Lab clips visible", "no key yet")
        return
    try:
        from lab.clips import load_manifest
        manifest = load_manifest()
    except Exception as exc:
        record(SKIP, "Lab clips visible", f"no manifest ({type(exc).__name__})")
        return
    clip = next((c for c in manifest.values() if c.uri), None)
    if clip is None:
        record(
            SKIP, "Lab clips visible", "manifest carries no URIs yet",
            "expected before the session: instructors upload the clips on the day",
        )
        return
    try:
        config.client().files.get(name="files/" + clip.uri.rsplit("/", 1)[-1])
        record(PASS, "Lab clips visible", f"{clip.clip_id} readable with this key")
    except Exception:
        record(
            WARN, "Lab clips visible", "this key cannot read the lab's clips",
            "expected before the day. On Tuesday it means either the clips "
            "have not been refreshed yet, or your key is from a different "
            "project than the one they were uploaded to. Tell the instructors "
            "rather than debugging it",
        )


def check_generation(enabled: bool) -> None:
    """Make one real, billed call. Instructors only.

    Participants never need this — everything above is free. It exists so
    whoever funds the keys can confirm one works end to end before handing it
    out, which is the check that catches an unfunded key.
    """
    if not enabled:
        record(SKIP, "Generation (billed)", "instructors: pass --spend to run this")
        return
    if not config.api_key():
        record(SKIP, "Generation (billed)", "no key to test with")
        return
    model = config.model()
    try:
        reply = config.client().models.generate_content(
            model=model, contents="Reply with OK."
        )
        text = (getattr(reply, "text", "") or "").strip()[:20]
        record(PASS, "Generation (billed)", f"{model} replied {text!r}")
    except Exception as exc:
        text = str(exc)
        if "RESOURCE_EXHAUSTED" in text and (
            "credit" in text.lower() or "billing" in text.lower()
        ):
            record(
                WARN,
                "Generation (billed)",
                f"{model} resolved; key has no credit",
                "the key needs funding before it is handed to a participant: "
                "https://aistudio.google.com/ (Billing)",
            )
            return
        record(
            FAIL,
            "Generation (billed)",
            f"{type(exc).__name__}: {str(exc)[:100]}",
            config.explain_api_error(exc),
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
        # One command, 335 KB, about a second. Only mention LAB_DATA_DIR for
        # the rarer case where they already have a copy elsewhere.
        "run  python -m lab.get_data  — or set LAB_DATA_DIR if you "
        "already have the labels somewhere else",
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


#: Extension ids the lab's run targets and completions depend on. All three are
#: on Open VSX, which is the only gallery Antigravity installs from. Pylance is
#: deliberately absent: Microsoft licenses it to official VS Code only, so it
#: cannot be installed here at all.
ANTIGRAVITY_EXTENSIONS = {
    "ms-python.python": "Python language support",
    "ms-python.debugpy": "the debugger the Run and Debug targets use",
    "detachhead.basedpyright": "completions, since Pylance is unavailable",
}


def _antigravity_extension_ids() -> set[str] | None:
    """Extension ids Antigravity has installed, or None if it is not installed.

    Read off disk rather than by launching the IDE: ``extensions.json`` in the
    user extensions directory is the same manifest the IDE maintains, and
    reading it costs nothing. ``Path.home()`` keeps this working on Windows,
    where the directory sits under the user profile.
    """
    root = Path.home() / ".antigravity-ide" / "extensions"
    if not root.is_dir():
        return None
    manifest = root / "extensions.json"
    ids: set[str] = set()
    if manifest.is_file():
        try:
            for entry in json.loads(manifest.read_text()):
                identifier = entry.get("identifier", {}).get("id")
                if identifier:
                    ids.add(identifier.lower())
        except (ValueError, OSError):
            pass  # fall through to the directory listing
    if not ids:
        # Directory names are `publisher.name-version-platform`.
        for child in root.iterdir():
            if child.is_dir() and "." in child.name:
                ids.add(child.name.rsplit("-", 2)[0].lower())
    return ids


def check_antigravity() -> None:
    """Antigravity and the three extensions the run targets need.

    Never a failure. Antigravity is the IDE the session assumes, but every part
    of this lab runs from a terminal, so a missing extension is something to
    fix before Tuesday rather than something that stops you working.
    """
    installed = _antigravity_extension_ids()
    if installed is None:
        record(
            WARN, "Antigravity", "not installed, or never opened",
            "install it and open this folder once, then re-run. Everything "
            "here also works from a terminal, so this is not a blocker",
        )
        return

    missing = [e for e in ANTIGRAVITY_EXTENSIONS if e.lower() not in installed]
    if not missing:
        record(PASS, "Antigravity extensions", f"all {len(ANTIGRAVITY_EXTENSIONS)} present")
        return
    record(
        WARN,
        "Antigravity extensions",
        f"{len(missing)} missing: {', '.join(missing)}",
        "open this folder in Antigravity and accept the recommended "
        "extensions, or install them from the Extensions panel. Needed for "
        + "; ".join(ANTIGRAVITY_EXTENSIONS[e] for e in missing),
    )


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
        "--spend",
        action="store_true",
        help="instructors only: also make one real, billed call to confirm a "
        "key is funded. Participants never need this.",
    )
    args = parser.parse_args()

    config.load_env()

    print(f"Surgical Agent Lab — preflight on {platform.platform()}\n")
    check_python()
    check_virtualenv()
    check_packages()
    check_api_key()
    check_api_reachable()
    check_dataset()
    check_pipeline_end_to_end()
    check_port()
    check_antigravity()
    check_clip_visibility()
    check_generation(args.spend)

    failed = [r for r in _results if r[0] == FAIL]
    skipped = [r for r in _results if r[0] == SKIP]
    warned = [r for r in _results if r[0] == WARN]
    print("\n" + "-" * 68)
    print("COPY EVERYTHING BELOW THIS LINE INTO YOUR REPLY")
    print("-" * 68)
    print(f"python   {sys.version.split()[0]}   platform {platform.platform()}")
    for status, name, detail, _ in _results:
        print(f"{status:5} {name:32} {detail}")
    passed = len(_results) - len(failed) - len(skipped) - len(warned)
    print(
        f"\n{passed} passed, {len(failed)} failed, "
        f"{len(warned)} needing attention, {len(skipped)} skipped"
    )
    if failed:
        print("Not ready. Apply the fixes above and re-run.")
    elif warned:
        print(
            "Setup is correct. See the note above for what still needs doing "
            "before the session."
        )
    elif skipped:
        print("Ready. The skipped check is the instructors' billed one.")
    else:
        print("Ready. Nothing further to do before the session.")
    print("-" * 68)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
