"""Getting a clip in front of the model, whichever key you are using.

SUPPLIED — you do not need to change this file.

The Gemini API cannot read ``gs://`` URIs, so a clip reaches the model through
the Files API: upload it, get a URI back, reference that. Two wrinkles make a
naive "just upload it" approach unreliable in a room of twenty-five people:

* **Uploaded files live for 48 hours.** A URI that worked yesterday may not
  work today.
* **Files belong to the project the key came from.** Instructors pre-upload the
  clips once so nobody spends lab time on it — but that only helps you if your
  key is from the same project. Bring your own key and those URIs are invisible
  to you.

:func:`resolve_clip` deals with both. It tries the pre-uploaded URI, checks it
is really there, and falls back to uploading your own copy if not. Your code
calls it and gets something that works::

    uri = resolve_clip("case_045_p1_213")

If neither route is available it raises with the specific reason, rather than
letting a confusing 403 surface from the middle of a Gemini call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lab.env import REPO_ROOT, client

#: Written by tools/upload_clips.py, committed so participants inherit the URIs.
MANIFEST_PATH = REPO_ROOT / "lab" / "clips.json"

#: Where local copies live, if they were distributed.
CLIP_DIR = REPO_ROOT / "data" / "clips"

# Resolved URIs, so a repeated call in one session does not re-upload.
_resolved: dict[str, str] = {}


class ClipUnavailable(RuntimeError):
    """A clip could not be reached by any route, with the reason in the message."""


@dataclass(frozen=True)
class Clip:
    """One window of footage the rules flagged.

    Attributes:
        clip_id: e.g. ``"case_045_p1_213"``.
        case_id: the session it came from.
        part: video part, because time restarts in each one.
        start_s: window start within that part.
        end_s: window end within that part.
        uri: pre-uploaded Files API URI, if an instructor uploaded it.
        local: path to a local copy, if one was distributed.
    """

    clip_id: str
    case_id: str
    part: int
    start_s: float
    end_s: float
    uri: str | None = None
    local: str | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def load_manifest() -> dict[str, Clip]:
    """Every clip the manifest knows about, keyed by clip id.

    Returns an empty mapping if there is no manifest yet, so the rest of the
    lab still runs.
    """
    if not MANIFEST_PATH.exists():
        return {}
    raw = json.loads(MANIFEST_PATH.read_text())
    out = {}
    for clip_id, entry in raw.get("clips", {}).items():
        out[clip_id] = Clip(
            clip_id=clip_id,
            case_id=entry.get("case_id", ""),
            part=int(entry.get("part", 1)),
            start_s=float(entry.get("start_s", 0.0)),
            end_s=float(entry.get("end_s", 0.0)),
            uri=entry.get("uri") or None,
            local=entry.get("local") or None,
        )
    return out


def _uri_is_reachable(uri: str) -> bool:
    """Whether this key can actually see that file.

    A URI from another project, or one whose 48 hours have run out, fails here
    rather than three layers deeper inside a generate call.
    """
    name = uri.rstrip("/").split("/files/")[-1]
    try:
        info = client().files.get(name=f"files/{name}")
    except Exception:
        return False
    return str(getattr(info, "state", "")).endswith("ACTIVE")


def _local_path(clip: Clip) -> Path | None:
    for candidate in (
        Path(clip.local) if clip.local else None,
        REPO_ROOT / (clip.local or ""),
        CLIP_DIR / f"{clip.clip_id}.mp4",
    ):
        if candidate and candidate.is_file():
            return candidate
    return None


def _upload(path: Path) -> str:
    """Upload one file and wait until the API has finished processing it."""
    import time

    handle = client().files.upload(file=str(path))
    deadline = time.time() + 120
    while str(handle.state).endswith("PROCESSING") and time.time() < deadline:
        time.sleep(2)
        handle = client().files.get(name=handle.name)
    if not str(handle.state).endswith("ACTIVE"):
        raise ClipUnavailable(
            f"{path.name} finished uploading in state {handle.state}, not ACTIVE"
        )
    return handle.uri


def resolve_clip(clip_id: str) -> str:
    """A Files API URI for this clip that the current key can actually use.

    Tries, in order: the pre-uploaded URI from the manifest, then uploading a
    local copy. Results are remembered for the rest of the session.

    Args:
        clip_id: e.g. ``"case_045_p1_213"``.

    Raises:
        ClipUnavailable: if no route works, naming which ones were tried.
    """
    if clip_id in _resolved:
        return _resolved[clip_id]

    manifest = load_manifest()
    clip = manifest.get(clip_id)
    if clip is None:
        known = ", ".join(sorted(manifest)[:5]) or "none"
        raise ClipUnavailable(
            f"{clip_id!r} is not in the manifest. Known clips: {known}…"
        )

    # 1 — the instructors' pre-upload, if this key can see it.
    if clip.uri and _uri_is_reachable(clip.uri):
        _resolved[clip_id] = clip.uri
        return clip.uri

    # 2 — our own copy.
    path = _local_path(clip)
    if path is not None:
        uri = _upload(path)
        _resolved[clip_id] = uri
        return uri

    raise ClipUnavailable(
        f"cannot reach {clip_id!r}.\n"
        f"  pre-uploaded URI: {'present but not visible to this key' if clip.uri else 'none in manifest'}\n"
        f"  local copy: not found under {CLIP_DIR}\n"
        "If you are using your own API key, the instructors' uploads belong to "
        "a different project and are invisible to you. Either switch to the key "
        "you were given, or put a copy of the clip in data/clips/."
    )


def clip_id_for(case_id: str, part: int, start_s: float) -> str:
    """The canonical id for the clip covering a flagged window.

    Clips are cut to the windows the rules flag, so a case, a part and a start
    time identify one exactly. Keeping the id derivable means callers do not
    have to carry clip ids around alongside the measurements.

    Args:
        case_id: e.g. ``"case_045"``.
        part: video part the window belongs to.
        start_s: window start, seconds within that part.
    """
    return f"{case_id}_p{int(part)}_{int(round(start_s))}"


def find_for_window(
    case_id: str, part: int, start_s: float, end_s: float
) -> Clip | None:
    """The manifest entry whose footage covers this window, if there is one.

    Tries the derived id first, then falls back to any clip in the same part
    that overlaps — clips are cut with a little padding, so a flag's window and
    its clip's boundaries rarely match to the second.

    Args:
        case_id: e.g. ``"case_045"``.
        part: video part the window belongs to.
        start_s: window start, seconds within that part.
        end_s: window end, seconds within that part.
    """
    manifest = load_manifest()
    exact = manifest.get(clip_id_for(case_id, part, start_s))
    if exact is not None:
        return exact
    overlapping = [
        c
        for c in manifest.values()
        if c.case_id == case_id
        and c.part == int(part)
        and c.start_s < end_s
        and c.end_s > start_s
    ]
    if not overlapping:
        return None
    # the clip that covers most of the window
    return max(
        overlapping,
        key=lambda c: min(c.end_s, end_s) - max(c.start_s, start_s),
    )


def clips_for_case(case_id: str) -> list[Clip]:
    """Manifest entries belonging to one session, in playback order.

    Args:
        case_id: e.g. ``"case_045"``.
    """
    found = [c for c in load_manifest().values() if c.case_id == case_id]
    return sorted(found, key=lambda c: (c.part, c.start_s))
