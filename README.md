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
python -m lab.get_data                              # the dataset: 335 KB, one second
python -m lab.set_key                               # paste your key when prompted
python preflight.py
```

That is the whole setup. **No gcloud, no cloud project, no sign-in** — the lab
authenticates with a single Gemini API key. Get one from
[aistudio.google.com/apikey](https://aistudio.google.com/apikey).

`lab.set_key` writes your key straight into `.env` from a hidden prompt, so it
never reaches your screen, your shell history or your clipboard manager. Edit
`.env` by hand instead if you would rather — `cp .env.example .env` gives you a
commented template.

`lab.get_data` downloads the published SurgVU label release from its own
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

`python -m lab.get_data` puts it at **`data/labels/`** inside this
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

Antigravity is the IDE for this session. Everything here also works from a
plain terminal, so nothing below is load-bearing — but the session assumes you
are in Antigravity, and these are the four things that trip people up.

**You sign in with your own account.** Antigravity's login is yours, not
something issued for the lab. The API key we send you is *not* a login: it goes
in `.env` and is what your code uses to call Gemini. Two Google-adjacent
credentials doing two unrelated jobs, and confusing them costs ten minutes.

**Trust the folder when it asks.** A freshly cloned repository opens in
Restricted Mode, which silently disables the debugger and the run targets. If
`Run and Debug` does nothing, this is why — click **Trust** in the banner, or
`File → Trust Folder`.

**Create the virtual environment before you open the folder**, or reload the
window afterwards (`Developer: Reload Window`). The interpreter is found by
auto-discovery rather than a hard-coded path, because a hard-coded one is wrong
on Windows. If it does not pick up `.venv`, run **`Python: Select Interpreter`**
and choose it.

**Take the three extension recommendations.** Antigravity installs from Open
VSX, so the ids in `.vscode/extensions.json` were checked against it:

| Extension | Why |
|---|---|
| `ms-python.python` | Python language support |
| `ms-python.debugpy` | the debugger every `Run and Debug` target uses |
| `detachhead.basedpyright` | completions |

Open the folder and accept the prompt, or install them from the Extensions
panel. **`python preflight.py` checks all three** and names any that are
missing, so you do not have to remember this.

**Pylance is not on Open VSX** — Microsoft licenses it to official VS Code
only, and Antigravity answers `Extension 'ms-python.vscode-pylance' not found`.
Do not go looking for it. basedpyright does that job here, with type checking
turned off so you see your own mistakes rather than its opinions about pandas.

Run targets under `Run and Debug`:

| | |
|---|---|
| Setup: get the dataset | `python -m lab.get_data` |
| Setup: set your API key | `python -m lab.set_key`, hidden prompt |
| Run the lab interface | Streamlit on `ui/app.py` |
| Preflight | `python preflight.py` |
| Evaluate the rules | what the rules find, per case — useful for the threshold stretch exercise |

The two setup targets exist so the whole path works without touching a
terminal — worth knowing if `source .venv/bin/activate` is where you get stuck.

### The agent, and one rule for it

`AGENTS.md` orients the IDE's agent: what is supplied, what you write, the
traps in the data, and the rules it must not break — chiefly that it must never
invent a number. Antigravity reads it at the workspace root (it also reads
`GEMINI.md` and `.agents/rules`, and shows each as its own source).

**Do not paste your API key into the agent chat.** The agent already reads your
workspace, including `.env`, so it has no need of the key quoted at it — and
pasted secrets end up in transcripts and commits, which outlive the key.

## Where the clips come from

The clips this repository references were cut from the published SurgVU video
archive — a single 344 GB zip — down to the 40-second window that explains each
flag. That pipeline is **not in this repository**: it reads raw video nobody
here needs, it writes the artefacts this repo already ships (`lab/clips.json`,
`lab/cohort.json`), and keeping it out means everything you can see is
something you might actually run.

You get the clips. You never need the source video, `ffmpeg`, or 11.6 GB of
disk.

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
| `preflight.py` | ten environment checks, none of which spends a token |

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
pytest -q            # 75 tests, no key needed and no tokens spent
```

The variant tabs in the interface stay dark until the matching file is
written, then come alive. Under every Coach answer is the list of tools it
actually called — the only way to tell a measured number from an invented one.

## Licence and the dataset

The code is MIT licensed — see [LICENSE](LICENSE). Clone it, change it, and
keep whatever you build on it; that includes after the session ends, because
your working copy is already a git repository and nothing about it depends on
credentials that expire.

**The dataset is not ours and is not covered by that licence.** `lab.get_data`
fetches the published SurgVU release from its own public bucket, so the
dataset's own terms apply to you directly. The clips the lab uses are cut from
the same release and are not redistributed here.

> **Not a medical device.** Retrospective review of recorded *training
> exercise* footage, for education. No diagnostic, intraoperative or patient-care
> claim is made or intended.

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
