"""Loading and normalising SurgVU session labels.

SUPPLIED — you do not need to change this file.

The raw labels have three properties that will bite you if you read them
naively, and this module deals with all three:

1. **Times are in two different formats.** ``tasks.csv`` stores seconds as
   floats; ``tools.csv`` stores ``HH:MM:SS.ffffff`` strings. Everything here
   is normalised to float seconds in ``start_s`` / ``end_s``.

2. **Long cases are split into parts, and time restarts at zero in each
   part.** Subtracting a ``stop_time`` in part 2 from a ``start_time`` in
   part 1 produces a negative duration. Every row therefore carries a
   ``part``, and any interval comparison must happen within a single part.
   Use :func:`overlapping_tools`, which enforces that for you.

3. **``tools.csv`` contains exact duplicate rows** — about a quarter of them
   across the corpus. They are dropped on load.

Typical use::

    case = load_case("case_045")
    for seg in case.tasks.itertuples():
        tools = overlapping_tools(case.tools, seg.part, seg.start_s, seg.end_s)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

# Where the label CSVs live. Override with LAB_DATA_DIR if your copy is
# somewhere else.
import os

DATA_DIR = Path(os.environ.get("LAB_DATA_DIR", "data/labels"))

_CLOCK = re.compile(r"^\s*(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*$")

# tools.csv rows are considered duplicates when all of these match.
_TOOL_DEDUPE_KEY = [
    "install_case_time",
    "uninstall_case_time",
    "arm",
    "groundtruth_toolname",
]


def to_seconds(value) -> float | None:
    """Convert a SurgVU timestamp to float seconds.

    Accepts either ``HH:MM:SS.ffffff`` (the ``tools.csv`` format) or a bare
    number of seconds (the ``tasks.csv`` format). Returns ``None`` for blanks
    and anything unparseable, so callers can drop bad rows rather than crash.

    Args:
        value: the raw cell contents.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = _CLOCK.match(str(value))
    if match:
        hours, minutes, seconds = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Case:
    """One recorded session: its task segments and its tool usage.

    Attributes:
        case_id: e.g. ``"case_045"``.
        tasks: columns ``part``, ``start_s``, ``end_s``, ``task``,
            ``duration_s`` — one row per labelled task segment.
        tools: columns ``part``, ``start_s``, ``end_s``, ``arm``, ``tool``,
            ``commercial`` — one row per instrument mount.
    """

    case_id: str
    tasks: pd.DataFrame
    tools: pd.DataFrame

    @property
    def parts(self) -> list[int]:
        """Video parts this case is split across, in order."""
        values = set(self.tasks["part"]) | set(self.tools["part"])
        return sorted(int(p) for p in values)

    @property
    def is_multipart(self) -> bool:
        """True when the case spans more than one video part."""
        return len(self.parts) > 1


def list_cases() -> list[str]:
    """Return every case id present in the data directory, sorted."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("case_")
    )


def load_tasks(case_id: str) -> pd.DataFrame:
    """Load and normalise the task segments for one case.

    Rows with an unparseable time or a missing task name are dropped — about
    a dozen exist across the corpus. Segments are returned in playback order.

    Args:
        case_id: e.g. ``"case_045"``.
    """
    raw = pd.read_csv(DATA_DIR / case_id / "tasks.csv")
    out = pd.DataFrame(
        {
            "part": raw.get("start_part"),
            "start_s": raw["start_time"].map(to_seconds),
            "end_s": raw["stop_time"].map(to_seconds),
            "task": raw["groundtruth_taskname"].astype("string").str.strip(),
        }
    )
    # A segment that changes part mid-way cannot be timed, so drop it.
    if "stop_part" in raw.columns:
        out = out[raw["start_part"].eq(raw["stop_part"]).fillna(False).values]

    out = out.dropna(subset=["part", "start_s", "end_s", "task"])
    out = out[out["task"].str.lower() != "nan"]
    out = out[out["end_s"] > out["start_s"]]
    out["part"] = out["part"].astype(int)
    out["duration_s"] = out["end_s"] - out["start_s"]
    return out.sort_values(["part", "start_s"]).reset_index(drop=True)


def load_tools(case_id: str) -> pd.DataFrame:
    """Load and normalise the tool usage for one case.

    Exact duplicate rows are dropped. Rows with an unparseable time, or that
    span a part boundary, are dropped.

    Args:
        case_id: e.g. ``"case_045"``.
    """
    raw = pd.read_csv(DATA_DIR / case_id / "tools.csv")
    present = [c for c in _TOOL_DEDUPE_KEY if c in raw.columns]
    if present:
        raw = raw.drop_duplicates(subset=present)

    out = pd.DataFrame(
        {
            "part": raw.get("install_case_part"),
            "start_s": raw["install_case_time"].map(to_seconds),
            "end_s": raw["uninstall_case_time"].map(to_seconds),
            "arm": raw["arm"].astype("string").str.strip(),
            "tool": raw["groundtruth_toolname"].astype("string").str.strip(),
            "commercial": raw.get(
                "commercial_toolname", pd.Series(dtype="string")
            ).astype("string"),
        }
    )
    if "uninstall_case_part" in raw.columns:
        same_part = (
            raw["install_case_part"].eq(raw["uninstall_case_part"]).fillna(False)
        )
        out = out[same_part.values]

    out = out.dropna(subset=["part", "start_s", "end_s"])
    out = out[out["end_s"] > out["start_s"]]
    out["part"] = out["part"].astype(int)
    # Blank and literal "nan" tool names are meaningful — a real instrument was
    # mounted and not identified — so they are kept and normalised to <unknown>.
    out["tool"] = out["tool"].fillna("<unknown>").replace(
        {"": "<unknown>", "nan": "<unknown>"}
    )
    out["duration_s"] = out["end_s"] - out["start_s"]
    return out.sort_values(["part", "start_s"]).reset_index(drop=True)


@lru_cache(maxsize=None)
def load_case(case_id: str) -> Case:
    """Load one case's tasks and tools, normalised and cached.

    Args:
        case_id: e.g. ``"case_045"``.
    """
    return Case(case_id, load_tasks(case_id), load_tools(case_id))


def overlapping_tools(
    tools: pd.DataFrame, part: int, start_s: float, end_s: float
) -> pd.DataFrame:
    """Tool mounts that overlap a time window **within the same part**.

    Always use this rather than comparing times directly: a mount in part 2
    and a segment in part 1 have unrelated clocks, and comparing them yields
    confident nonsense.

    Args:
        tools: a normalised tool frame, from :func:`load_tools`.
        part: the video part the window belongs to.
        start_s: window start, seconds within that part.
        end_s: window end, seconds within that part.
    """
    return tools[
        (tools["part"] == part) & (tools["start_s"] < end_s) & (tools["end_s"] > start_s)
    ]


def tool_changes_within(
    tools: pd.DataFrame, part: int, start_s: float, end_s: float
) -> pd.DataFrame:
    """Tool mounts that *begin* inside a window, within the same part.

    A mount that began earlier and merely continues through the window is not
    a change, so this is stricter than :func:`overlapping_tools`.

    Args:
        tools: a normalised tool frame, from :func:`load_tools`.
        part: the video part the window belongs to.
        start_s: window start, seconds within that part.
        end_s: window end, seconds within that part.
    """
    return tools[
        (tools["part"] == part)
        & (tools["start_s"] > start_s)
        & (tools["start_s"] < end_s)
    ]
