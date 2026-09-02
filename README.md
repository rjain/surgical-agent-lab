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
python tools/fetch_labels.py                        # the dataset: 335 KB, one second
cp .env.example .env                                # then paste your API key into it
python preflight.py
```

That is the whole setup. **No gcloud, no cloud project, no sign-in** — the lab
authenticates with a single Gemini API key. Get one from
[aistudio.google.com/apikey](https://aistudio.google.com/apikey).

`fetch_labels.py` downloads the published SurgVU label release from its own
public bucket, so the dataset's terms apply to you directly and nothing has
been repackaged in between. It is labels only — task segments and instrument
mounts as CSV. The video clips the lab needs are handled by `lab/clips.py`.

`preflight.py` prints a report to send back to the instructors. Run it when you
get your key, not on the day.

> **Nothing in the pre-flight spends a token.** The one check that touches the
> API lists the available models, which is free — enough to prove your network
> reaches Google, that corporate TLS inspection has not broken the SDK, that
> your key works, and that the model the lab uses is offered to it.
>
> Instructors funding the keys use `preflight.py --spend` to make one real
> billed call and confirm a key has credit. You do not need that.

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

`python tools/fetch_labels.py` puts it at **`data/labels/`** inside this
repository — so that `data/labels/case_045/tasks.csv` exists — and nothing
needs configuring.

If you already keep a copy elsewhere, set `LAB_DATA_DIR` either in a `.env`
file (copy `.env.example`) or as an environment variable. The lookup order is:

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
| Preflight (`--spend`) | instructors only: one billed call, to confirm a key is funded |
| Evaluate the rules | what the rules find, per case |

`AGENTS.md` orients the IDE's agent: what is supplied, what you write, and the
rules it must not break — chiefly that it must never invent a number.

## Preparing the data (instructors only)

The published SurgVU archive is a single 344 GB zip, but it is a public Cloud
Storage object and it stores one video per case — so only the cases we use need
downloading, about 11.6 GB for six.

```bash
pip install imageio-ffmpeg          # bundles ffmpeg; no system install needed
python tools/fetch_video.py --list  # 155 cases, curated six marked
python tools/fetch_video.py --curated
python tools/cut_clips.py           # 35 clips, ~45 MB, writes lab/clips.json
python tools/upload_clips.py        # Files API pre-upload, on the day
```

Participants never run any of this — `tools/fetch_labels.py` is the one tool
they do run. They get the clips, not the source video.

## What is supplied, and what you write

Everything below the model is done. Detection is finished and tested; you spend
the lab on the two layers above it.

| Supplied | |
|---|---|
| `lab/data.py` | parts-aware loading, dedupe, unit conversion |
| `lab/metrics.py` | per-step measurements, corpus comparison |
| `lab/rules.py` | the deterministic deviation engine — Lab 1, complete |
| `lab/cohort.json` | corpus medians, built from all 155 cases |
| `lab/clips.py` | gets a clip to the model, whichever key you hold |
| `lab/cache.py` | disk cache, so re-running a window is free |
| `lab/runtime.py` | drives an ADK session and records every tool call |
| `ui/app.py` | timeline, flagged moments, and both variant tabs |
| `preflight.py` | eight environment checks, none of which spends a token |

| You write | Lab |
|---|---|
| `lab/lab2_analyze.py` | 2 — the Gemini call, and the guardrail on its output |
| `lab/lab3_agent.py` | 3 — the tool docstrings, and one agent |
| `lab/variants/coach.py` | Variant A — an agent you talk to |
| `lab/variants/auditor.py` | Variant B — an agent you launch and leave |

Each of those raises `NotImplementedError` with what it wants in the docstring.
A reference implementation of every one is in `solutions/`; reading it after
you have tried is a normal move.

```bash
pytest -q            # 36 tests, no key needed and no tokens spent
```

The variant tabs in the interface stay dark until the matching file is
written, then come alive. Under every Coach answer is the list of tools it
actually called — the only way to tell a measured number from an invented one.

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
