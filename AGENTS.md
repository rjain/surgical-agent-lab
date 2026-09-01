# Orientation for coding agents

Context for any AI assistant working in this repository — Antigravity's agent,
or anything else. Read this before making changes.

## What this is

A teaching repository. Participants build a system that reviews recorded
da Vinci **training exercise** sessions and flags moments worth discussing.
Three labs, in order:

| Lab | File | Who writes it |
|---|---|---|
| 1 — Detection | `lab/data.py`, `lab/metrics.py`, `lab/rules.py` | **Supplied, tested. Do not modify.** |
| 2 — Explanation | `lab/cp1_analyze.py` | The participant |
| 3 — Application | `lab/cp2_agent.py`, `lab/variants/*.py` | The participant |

`ui/app.py` and `preflight.py` are supplied and working.

## Rules that are not style preferences

**Never invent a number.** Every measurement comes from `lab/metrics.py` or
`lab/rules.py`. If code or prose states a duration, a count or a ratio, it must
be traceable to one of those. This is the whole point of the exercise, and an
agent that helpfully estimates a plausible figure defeats it.

**This is not a medical device.** The vocabulary matters. Deviations carry a
`score`, never a `severity` or a `risk`. These are efficiency observations
about a recorded training exercise, not clinical findings, and nothing here
may be phrased as diagnosis, intraoperative guidance or patient care.

**Do not modify the supplied Lab 1 modules.** They are tested and the tests
guard real properties of the data. Tuning the thresholds at the top of
`lab/rules.py` is a legitimate exercise; changing the loaders is not.

**No model in the detection path.** `lab/rules.py` is ordinary Python and must
stay that way. Layer 2 explains what Layer 1 already flagged; it never decides
what to flag.

## Three traps in the raw data

`lab/data.py` handles all three. If you write new code that touches the CSVs
directly, you have to handle them yourself — so prefer the loaders.

1. **Time restarts at zero in each video part.** Subtracting a time in part 2
   from one in part 1 yields a negative duration that looks plausible. Use
   `overlapping_tools()` and `tool_changes_within()`, which enforce a single
   part.
2. **`tools.csv` contains exact duplicate rows** — roughly a quarter of the
   corpus. Dropped on load.
3. **Two time formats.** `tasks.csv` stores float seconds; `tools.csv` stores
   `HH:MM:SS.ffffff`. Everything is normalised to `start_s` / `end_s`.

## Conventions

- Docstrings are the tool contract. ADK builds the declaration the model sees
  from the signature and the docstring, so vague docstrings produce agents that
  pick the wrong tool. Write them as an interface, with an `Args:` block.
- Pinned dependencies. Do not upgrade anything in `requirements.txt`.
- Tests live in `tests/` and run with `pytest -q`. They must stay green.

## Checking your work

```bash
pytest -q                          # 16 tests over the supplied modules
python preflight.py                # environment checks
python tools/evaluate_rules.py     # what the rules find, per case
streamlit run ui/app.py            # the interface
```

`LAB_DATA_DIR` must point at the folder holding the `case_*` directories.
