"""Precompute corpus-wide statistics into lab/cohort.json.

Run once when the dataset changes. Participants never run this; it exists so
that metrics.py does not have to read 155 cases at import time.
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.data import list_cases, load_case  # noqa: E402


def main() -> None:
    durations: dict[str, list[float]] = {}
    cases = list_cases()
    for case_id in cases:
        try:
            case = load_case(case_id)
        except Exception as exc:  # a malformed case must not stop the build
            print(f"  skipped {case_id}: {exc}", file=sys.stderr)
            continue
        for row in case.tasks.itertuples():
            durations.setdefault(row.task, []).append(row.duration_s)

    payload = {
        "source_cases": len(cases),
        "median_duration_s": {
            k: round(statistics.median(v), 1) for k, v in sorted(durations.items()) if v
        },
        "segment_count": {k: len(v) for k, v in sorted(durations.items())},
    }
    out = Path(__file__).resolve().parents[1] / "lab" / "cohort.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out} from {len(cases)} cases, {len(payload['median_duration_s'])} steps")


if __name__ == "__main__":
    main()
