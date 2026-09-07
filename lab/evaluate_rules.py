#!/usr/bin/env python3
"""Report what the rules engine finds, so thresholds can be tuned on evidence.

Re-run this whenever a threshold in ``lab/rules.py`` changes, or when the
curated subset is revisited. A rule that never fires is dead weight in the
room; a rule that fires on the majority case teaches people the system cries
wolf. This script is how you tell the difference.

    python -m lab.evaluate_rules              # the curated subset
    python -m lab.evaluate_rules --all        # every case in the corpus
    python -m lab.evaluate_rules --candidates # best cases to curate
    python -m lab.evaluate_rules --distribution # per-rule rates against chance

Uses the same dataset the rest of the lab does; run `python -m lab.get_data`
first if you have not already.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path


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



def print_distribution() -> None:
    """Per-rule fire rates against the right denominator, with a null model.

    Counting flags does not tell you whether a rule is any good. Two rules can
    fire at the same rate while one detects real structure and the other slices
    a percentile off a smooth distribution and calls it an anomaly. The
    question worth asking is whether a rule beats chance on the same data.
    """
    import random
    import statistics

    import pandas as pd

    from lab.metrics import step_metrics
    from lab.rules import MIN_SEGMENT_S, OVERRUN_RATIO, SWAP_CHANGES

    cases = []
    for case_id in sorted(list_cases()):
        try:
            cases.append(step_metrics(load_case(case_id)))
        except Exception as exc:  # a malformed case should not stop the report
            print(f"  skipped {case_id}: {type(exc).__name__}: {exc}", file=sys.stderr)

    segments = [row for m in cases for row in m.itertuples()]
    eligible = [
        r for r in segments
        if r.duration_ratio is not None
        and not pd.isna(r.duration_ratio)
        and r.duration_s >= MIN_SEGMENT_S
    ]

    # Oscillation can only fire from the third segment of a part onwards, so it
    # gets a smaller denominator than the rules that look at every segment.
    osc_fires = osc_positions = 0
    for m in cases:
        tasks, parts = list(m["task"]), list(m["part"])
        for i in range(2, len(tasks)):
            osc_positions += 1
            if (tasks[i] == tasks[i - 2] and tasks[i] != tasks[i - 1]
                    and parts[i] == parts[i - 2]):
                osc_fires += 1

    rates = {
        "swap_rate": (sum(1 for r in segments if r.tool_changes >= SWAP_CHANGES),
                      len(segments)),
        "step_overrun": (sum(1 for r in eligible if r.duration_ratio >= OVERRUN_RATIO),
                         len(eligible)),
        "step_oscillation": (osc_fires, osc_positions),
        "unknown_instrument": (sum(1 for r in segments if r.has_unknown_tool),
                               len(segments)),
    }

    print(f"Rule behaviour across {len(cases)} sessions, {len(segments)} segments.\n")
    print(f"{'rule':20} {'fires':>6} {'of':>8} {'rate':>6}")
    print("-" * 44)
    for rule in RULE_IDS:
        fires, denominator = rates[rule]
        share = 100 * fires / denominator if denominator else 0.0
        print(f"{rule:20} {fires:6d} {denominator:8d} {share:5.0f}%")

    print("\n--- does the rule beat chance? ---\n")

    random.seed(0)
    trials = []
    for _ in range(300):
        total = 0
        for m in cases:
            tasks, parts = list(m["task"]), list(m["part"])
            if len(tasks) < 3:
                continue
            shuffled = tasks[:]
            random.shuffle(shuffled)
            for i in range(2, len(shuffled)):
                if (shuffled[i] == shuffled[i - 2] and shuffled[i] != shuffled[i - 1]
                        and parts[i] == parts[i - 2]):
                    total += 1
        trials.append(total)
    chance = statistics.mean(trials)
    low, high = sorted(trials)[7], sorted(trials)[292]
    print("step_oscillation  task labels shuffled within each session, A-B-A recounted")
    print(f"                  observed {osc_fires}, chance {chance:.0f} "
          f"(95% range {low}-{high}) -> {osc_fires / chance:.2f}x")

    per_case = []
    for m in cases:
        e = [r for r in m.itertuples()
             if r.duration_ratio is not None and not pd.isna(r.duration_ratio)
             and r.duration_s >= MIN_SEGMENT_S]
        if len(e) >= 3:
            per_case.append((len(e), sum(1 for r in e if r.duration_ratio >= OVERRUN_RATIO)))
    base = sum(h for _, h in per_case) / sum(n for n, _ in per_case)
    observed_var = statistics.variance([h / n for n, h in per_case])
    simulated = [
        statistics.variance([sum(random.random() < base for _ in range(n)) / n
                             for n, _ in per_case])
        for _ in range(300)
    ]
    chance_var = statistics.mean(simulated)
    print("\nstep_overrun      do overruns cluster in particular sessions, or scatter")
    print("                  like the tail of a distribution?")
    print(f"                  observed variance {observed_var:.4f}, "
          f"chance {chance_var:.4f} -> {observed_var / chance_var:.2f}x")

    print("\n--- where would you cut? ---\n")
    ratios = sorted(r.duration_ratio for r in eligible)
    for threshold in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4):
        share = 100 * sum(1 for x in ratios if x >= threshold) / len(ratios)
        here = "   <- OVERRUN_RATIO" if abs(threshold - OVERRUN_RATIO) < 1e-9 else ""
        print(f"  duration_ratio >= {threshold:.1f}x : {share:5.1f}% of segments{here}")
    print("\n  A threshold on a smooth distribution is a percentile, not a")
    print("  detection. Look for a gap before you pick a number.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="every case in the corpus")
    group.add_argument(
        "--distribution",
        action="store_true",
        help="per-rule fire rates against a null model, to judge rule quality",
    )
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

    if args.distribution:
        print_distribution()
        return 0

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
