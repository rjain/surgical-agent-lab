#!/usr/bin/env python3
"""Report what the rules engine finds, so thresholds can be tuned on evidence.

Re-run this whenever a threshold in ``lab/rules.py`` changes, or when the
curated subset is revisited. A rule that never fires is dead weight in the
room; a rule that fires on the majority case teaches people the system cries
wolf. This script is how you tell the difference.

    python tools/evaluate_rules.py              # the curated subset
    python tools/evaluate_rules.py --all        # every case in the corpus
    python tools/evaluate_rules.py --candidates # best cases to curate

Requires LAB_DATA_DIR to point at the label directory.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.data import list_cases, load_case  # noqa: E402
from lab.rules import RULES, find_deviations  # noqa: E402

#: The cases prepared for the lab. Chosen for rule coverage rather than size —
#: see the table this script prints.
CURATED = {
    "case_045": "golden path — single part, all 7 steps, written walkthrough",
    "case_129": "second clean case — richest segment count",
    "case_125": "all four rules on a simple single-part case",
    "case_036": "the messy one — two parts, widest tool set",
    "case_044": "unknown-instrument heavy — the log does not know",
    "case_059": "the quiet one — few flags, and that is the point",
}

RULE_IDS = ["swap_rate", "step_overrun", "step_oscillation", "unknown_instrument"]
SHORT = {
    "swap_rate": "swap",
    "step_overrun": "over",
    "step_oscillation": "osc",
    "unknown_instrument": "unk",
}


def summarise(case_id: str) -> dict:
    case = load_case(case_id)
    found = find_deviations(case_id)
    by_rule = collections.Counter(d.rule_id for d in found)
    return {
        "case": case_id,
        "parts": len(case.parts),
        "segments": len(case.tasks),
        "tasks": int(case.tasks["task"].nunique()),
        "flags": len(found),
        "kinds": sum(1 for r in RULE_IDS if by_rule[r]),
        "by_rule": by_rule,
    }


def print_table(rows: list[dict], note: dict[str, str] | None = None) -> None:
    header = (
        f"{'case':10} {'parts':>5} {'segs':>5} {'steps':>5} {'flags':>5} "
        f"{'cover':>6}  " + " ".join(f"{SHORT[r]:>4}" for r in RULE_IDS)
    )
    print(header)
    print("-" * (len(header) + (30 if note else 0)))
    totals = collections.Counter()
    steps: set[str] = set()
    for row in rows:
        totals.update(row["by_rule"])
        cover = "".join("X" if row["by_rule"][r] else "." for r in RULE_IDS)
        line = (
            f"{row['case']:10} {row['parts']:5d} {row['segments']:5d} "
            f"{row['tasks']:5d} {row['flags']:5d} {cover:>6}  "
            + " ".join(f"{row['by_rule'][r]:4d}" for r in RULE_IDS)
        )
        if note and row["case"] in note:
            line += f"   {note[row['case']]}"
        print(line)

    print("-" * len(header))
    print(
        f"{'TOTAL':10} {'':5} {sum(r['segments'] for r in rows):5d} {'':5} "
        f"{sum(r['flags'] for r in rows):5d} {'':6}  "
        + " ".join(f"{totals[r]:4d}" for r in RULE_IDS)
    )
    dead = [r for r in RULE_IDS if not totals[r]]
    print()
    if dead:
        print(f"  WARNING: these rules never fired: {', '.join(dead)}")
    else:
        print("  every rule fires at least once")
    print(f"  clip prep: {sum(r['flags'] for r in rows)} flag windows, "
          f"{len(rows)} whole-session passes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="every case in the corpus")
    group.add_argument(
        "--candidates",
        action="store_true",
        help="rank all cases by rule coverage, to help pick a curated subset",
    )
    parser.add_argument("cases", nargs="*", help="specific case ids to report on")
    args = parser.parse_args()

    available = set(list_cases())
    if not available:
        print("No cases found. Set LAB_DATA_DIR to the folder holding case_* dirs.")
        return 1

    if args.cases:
        targets, note = args.cases, None
    elif args.all or args.candidates:
        targets, note = sorted(available), None
    else:
        targets, note = [c for c in CURATED if c in available], CURATED
        missing = [c for c in CURATED if c not in available]
        if missing:
            print(f"note: curated cases absent from this dataset: {missing}\n")

    rows = []
    for case_id in targets:
        try:
            rows.append(summarise(case_id))
        except Exception as exc:
            print(f"  skipped {case_id}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if args.candidates:
        rows.sort(key=lambda r: (-r["kinds"], -r["tasks"], -r["flags"]))
        print("Cases ranked by rule coverage, then step coverage, then flag count.")
        print("Good subsets mix full-coverage cases with one quiet one.\n")
        print_table(rows[:20])
        print()
        zero = [r["case"] for r in rows if r["flags"] == 0]
        print(f"  {len(zero)} cases produce no flags at all"
              + (f" (e.g. {', '.join(zero[:5])})" if zero else ""))
        return 0

    if args.all:
        rows.sort(key=lambda r: -r["flags"])
        print(f"All {len(rows)} cases, most flags first.\n")

    print_table(rows, note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
