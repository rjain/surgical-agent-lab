# Surgical Workflow Deviation Auditor — starter repository

Review recorded da Vinci training-exercise sessions and find the moments worth
discussing. Built in three labs.

> **Not a medical device.** An educational exercise in retrospective review of
> recorded *training exercise* footage. Not for clinical use; no diagnostic or
> intraoperative claim.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # then paste your API key into it
python preflight.py --cloud
```

That is the whole setup. **No gcloud, no cloud project, no sign-in** — the lab
authenticates with a single Gemini API key. Get one from
[aistudio.google.com/apikey](https://aistudio.google.com/apikey).

`preflight.py` prints a report to send back to the instructors. Run it when you
get your key, not on the day.

> **The key needs credit on it.** A key with none looks fine everywhere until
> the first real call, which fails with a `429` about prepayment credits.
> `preflight.py --cloud` is what catches that, which is why it is worth running
> at least once.

### Settings

`.env` at the repository root, read by the code itself rather than by the
editor — so it behaves the same from the terminal, the debugger, Streamlit and
pytest. Both `KEY=value` and `export KEY="value"` work.

| | |
|---|---|
| `LAB_GEMINI_API_KEY` | required |
| `LAB_DATA_DIR` | only if the dataset is not at `data/labels/` |
| `LAB_MODEL` | defaults to `gemini-flash-latest` |

### Where the data goes

Put the dataset at **`data/labels/`** inside this repository — so that
`data/labels/case_045/tasks.csv` exists — and nothing needs configuring.

If you keep it elsewhere, set `LAB_DATA_DIR` either in a `.env` file (copy
`.env.example`) or as an environment variable. The lookup order is:

1. `LAB_DATA_DIR`, from the real environment or from `.env`
2. `data/labels` inside this repository
3. `data/labels` relative to wherever you ran the command

`lab/data.py` reads `.env` itself rather than relying on the editor. Editors
disagree about this — VS Code and Antigravity load `.env` for Run and Debug but
**not** for their integrated terminal, and Streamlit never reads one — so doing
it in code means the setting behaves the same however you start things. A real
environment variable still wins over the file.

## Opening this in Antigravity

Antigravity is the IDE for this session. **Sign in via `Business account` →
`Continue with Google Cloud` using your training account** — the default
sign-in path expects a personal Gmail address and will suggest you use one.
Do not: your training account is not a Gmail account, and moving the lab onto a
personal account defeats the disposable environment.

Then `File → Open Folder` on this directory. `.vscode/` is included, so the
interpreter, test discovery and run targets are picked up automatically once
the virtual environment exists — create it *before* opening the folder, or
reload the window afterwards.

Run targets under `Run and Debug`:

| | |
|---|---|
| Run the lab interface | Streamlit on `ui/app.py` |
| Preflight | local checks only |
| Preflight (`--cloud`) | adds one real model call |
| Evaluate the rules | what the rules find, per case |

`AGENTS.md` orients the IDE's agent: what is supplied, what you write, and the
rules it must not break — chiefly that it must never invent a number.

## Status

| | |
|---|---|
| `lab/data.py` | **supplied, tested** — parts-aware loading, dedupe, unit conversion |
| `lab/metrics.py` | **supplied, tested** — per-step measurements, corpus comparison |
| `lab/rules.py` | **supplied, tested** — the deterministic deviation engine |
| `lab/cohort.json` | corpus medians, built from 155 cases by `tools/build_cohort.py` |
| `preflight.py` | **working** — ten environment checks |
| `lab/cp1_analyze.py` | not yet written — Lab 2, the Gemini call |
| `lab/cp2_agent.py` | not yet written — tools and a single agent |
| `lab/variants/` | not yet written — Coach and Auditor |
| `ui/app.py` | not yet written |

```bash
pytest -q            # 16 tests over the supplied modules
```

## Three traps in the raw labels

`data.py` handles all three; the tests guard them.

1. **Times restart in each video part.** Subtracting across a part boundary
   gives a negative duration that looks plausible. Use `overlapping_tools()`.
2. **`tools.csv` has exact duplicate rows** — about a quarter of the corpus.
3. **Two time formats.** `tasks.csv` in float seconds, `tools.csv` in
   `HH:MM:SS.ffffff`. Everything is normalised to `start_s` / `end_s`.

## What the rules look for

Measured against all 155 cases before being chosen. An obvious candidate —
"an unexpected instrument for this step" — was rejected: monopolar scissors are
mounted during 98% of Uterine horn segments, so flagging that flags the norm.

| Rule | Fires when |
|---|---|
| `swap_rate` | 4+ instrument changes begin inside one step |
| `step_overrun` | a step runs 1.4x its corpus median |
| `step_oscillation` | a step is returned to after another ran in between |
| `unknown_instrument` | something was mounted but not identified |

Roughly four to five flags per session. Tuning the thresholds in `rules.py` is
a stretch exercise.
