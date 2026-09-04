"""The console's own metric vocabulary.

SUPPLIED — you do not need to change this file.

The customer publishes Objective Performance Indicators (OPI) from the
console: 109 named metrics, each with a display name and a definition. This
module carries the names and the definitions and nothing else.

It carries no values, deliberately. The sample we were given is dummy data,
and only the vocabulary was kept, so no measurement in this lab can come from
it. Numbers here are computed from the labels or they do not exist.

Only three of the 109 are computable from SurgVU labels: ``duration``,
``duration_tool`` and ``duration_armxtool``. The rest need console telemetry,
such as forces, clutches, pedal presses and instrument path length, that the
labels do not contain. Asking for one of those is not a failure. Saying what
it is and why it cannot be answered is the correct behaviour.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

#: Ships beside this module, like lab/cohort.json and lab/clips.json.
DICTIONARY_PATH = Path(__file__).with_name("metric_dictionary.csv")

#: The only columns ever read. Anything else in the file is ignored, which is
#: what keeps dummy values out of the lab.
_COLUMNS = ["metric_name", "display_name", "description", "derivable", "computed_from"]


@lru_cache(maxsize=1)
def metric_dictionary() -> dict[str, dict]:
    """Every console metric, keyed by its backend name.

    Each entry has ``metric_name``, ``display_name``, ``description``,
    ``derivable`` (a bool) and ``computed_from``.
    """
    frame = pd.read_csv(DICTIONARY_PATH, usecols=_COLUMNS)
    entries = {}
    for row in frame.itertuples():
        entries[row.metric_name] = {
            "metric_name": row.metric_name,
            "display_name": row.display_name,
            "description": row.description,
            "derivable": str(row.derivable).strip().lower() == "yes",
            "computed_from": ""
            if pd.isna(row.computed_from)
            else str(row.computed_from),
        }
    return entries


def describe(metric_name: str) -> dict | None:
    """What the console calls this metric and what it means.

    Args:
        metric_name: a backend metric name, e.g. ``"cam_clutch_cnt"``.

    Returns:
        The dictionary entry, or ``None`` if the console does not publish a
        metric by that name.
    """
    return metric_dictionary().get(metric_name)


def is_derivable(metric_name: str) -> bool:
    """Whether this metric can be computed from SurgVU labels.

    Args:
        metric_name: a backend metric name, e.g. ``"duration_tool"``.
    """
    entry = describe(metric_name)
    return bool(entry and entry["derivable"])
