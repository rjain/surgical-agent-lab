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
| 2 — Explanation | `lab/lab2_analyze.py` | The participant |
| 3 — Application | `lab/lab3_agent.py` | The participant |
| Variant A | `lab/variants/coach.py` | The participant — an agent you talk to |
| Variant B | `lab/variants/auditor.py` | The participant — an agent you launch and leave |

Groups pick **one** variant, not both. Each has a stated minimum shippable
form — a Coach without its Tracker, an Auditor as a plain loop — so a group
running late degrades its variant rather than losing it.

Supplied and working: `ui/app.py`, `preflight.py`, `lab/clips.py`,
`lab/cache.py`, `lab/config.py`, `lab/runtime.py`, `lab/get_data.py`,
`lab/set_key.py`.

`solutions/` holds a verified reference implementation of all four participant
files. **Do not copy one into `lab/` unless the participant asks.** Working out
that they are stuck is not the same as being asked to finish it for them, and
`tests/test_supplied.py` has a test that fails if a solution is committed into
`lab/`.

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
stay that way. Lab 2 explains what Lab 1 already flagged; it never decides
what to flag.

## Traps that have already cost real time

Each of these was hit for real. They are cheap to reintroduce and expensive to
notice, because every one of them fails *plausibly*.

**Send the watch window, never the whole segment.** A flagged segment is a
whole task step — median 11 minutes, up to 45 — and costs roughly 170,000
tokens. Every `Deviation` carries `watch_window`: 40 seconds around the instant
that explains the flag, about 10,000 tokens. That 16x saving is what keeps
twenty-five people inside a shared rate limit, and losing it is silent. The UI
shipped with exactly this bug for a while.

**`analyze_clip` takes `(case_id, part, t_start, t_end)`.** Passing
`(case_id, start_s, end_s)` puts a timestamp in the `part` slot. Python accepts
it, the clip resolves to the wrong part, and the window silently becomes the
whole clip. Time restarts at zero in each part, so a window means nothing
without one.

**No `tuple` in any schema the API sees.** Pydantic renders `tuple[float, float]`
as `prefixItems`, which the Gemini API rejects outright — before any network
call, with a message that does not mention tuples. Use two named floats.
`tests/test_supplied.py` walks the schema keys to guard this.

**Disable automatic function calling on direct `generate_content` calls.** The
SDK enables AFC by default and logs a warning even when no tools are passed.
Pass `automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)`.
Nothing in Lab 2 should call a tool — that is Lab 3's job — so saying so is the
honest fix rather than muting the log.

**`GOOGLE_GENAI_USE_ENTERPRISE`, not `GOOGLE_GENAI_USE_VERTEXAI`.** The old
name followed the Vertex AI → Gemini Enterprise rename and both `google-genai`
and ADK now emit a `DeprecationWarning` for it. `lab/config.py` sets only the
new name and removes the old one — setting both makes `google-genai` warn again
if they ever disagree. Every tutorial still shows the old name.

**ADK 2.8 puts tool parameters in `parameters_json_schema`.** The
`parameters` field on a `FunctionDeclaration` is `None`, so code reaching for
it gets an `AttributeError` on `None`. This is the `JSON_SCHEMA_FOR_FUNC_DECL`
experimental feature the warning is about.

**Clamp video offsets to the clip's duration.** An offset past the end returns
a bare `400 INVALID_ARGUMENT` that names nothing.

**A skeleton *defines* its functions and raises from inside them.** So a
successful `import` proves nothing about whether a participant has written
anything. Guarding on the import is why the Coach tab took the whole app down
in the state the repo ships in. Call the function and catch
`NotImplementedError`; treat both outcomes as expected states.

**`from __future__ import annotations` turns annotations into strings.** Code
that reads `__annotations__` to find a return type gets `"TechniqueNotes"`, not
the class. Use `typing.get_type_hints`. This is why `lab/cache.py` returned
dicts instead of models.

**`lab/config.py` *assigns* `GOOGLE_API_KEY`, it does not `setdefault`.**
Deferring to whatever is already in the environment is what lets a
participant's own key silently win. The lab key must win inside this process.

**`config.client()` is `lru_cache`d, deliberately.** A fresh client per call
closes the shared transport, and the next call fails with "Cannot send a
request, as the client has been closed."

## Three traps in the raw data

`lab/data.py` handles all three. If you write new code that touches the CSVs
directly, you have to handle them yourself — so prefer the loaders.

1. **Time restarts at zero in each video part.** Subtracting a time in part 2
   from one in part 1 yields a negative duration that looks plausible.
   `case_036` produced −172.8 minutes this way. Use `overlapping_tools()` and
   `tool_changes_within()`, which enforce a single part.
2. **`tools.csv` contains exact duplicate rows** — roughly a quarter of the
   corpus. Dropped on load.
3. **Two time formats.** `tasks.csv` stores float seconds; `tools.csv` stores
   `HH:MM:SS.ffffff`. Everything is normalised to `start_s` / `end_s`.

## Conventions

- Docstrings are the tool contract. ADK builds the declaration the model sees
  from the signature and the docstring, so vague docstrings produce agents that
  pick the wrong tool. Write them as an interface, with an `Args:` block.
  `get_metrics` in `lab/lab3_agent.py` ships with a **deliberately vague**
  docstring — that is the lesson, not an oversight. Do not fix it unprompted.
- Wire a helper agent as an **`AgentTool`**, not as a `sub_agent`. Transfer
  hands the conversation over; AgentTool borrows a capability. Most ADK
  material shows transfer first, and it is the wrong shape for the Coach.
- Show the tool-call trace under every agent answer. Prose reads identically
  whether a number came from a measurement or from nowhere; the trace is the
  only way to tell from outside. `lab/runtime.py` collects it.
- Pinned dependencies. Do not upgrade anything in `requirements.txt`.
- Authentication is a **Gemini API key** in `.env`, read through `lab/config.py`.
  There is no cloud project, no ADC and no endpoint region. Do not reintroduce
  them.
- **Never quote the API key back to the user, print it, or write it into a
  file.** It lives in `.env`, which is gitignored and chmod 0600. Do not paste
  it into a chat or a commit: transcripts and history outlive the key.
  **Do not write masking code either.** A regex that scanned `.env` and skipped
  a commented-out line has already leaked one key into a transcript, and that
  key had to be revoked. The one exception is deliberate and already written:
  `preflight.py` reports `key[-4:]` so that instructors handing out
  twenty-five individually-issued keys can tell which one a participant is on.
  That masks the *resolved value* rather than pattern-matching a file, which is
  why it is safe. Do not extend it, and do not add a second one.
- Video reaches the model through `lab.clips.resolve_clip()`, never a `gs://`
  URI — the Gemini API rejects those. Do not hand-roll uploads: uploaded files
  expire after 48 hours and belong to one project, and `resolve_clip` already
  handles both.
- Participant entry points are modules, all of them: `python -m lab.get_data`,
  `python -m lab.set_key`, `python -m lab.evaluate_rules`. There is no
  `tools/` directory — the instructor pipeline that builds the clips and
  `lab/cohort.json` lives in a separate private repository, deliberately.
- Tests live in `tests/` and run with `pytest -q`. They must stay green.

## Checking your work

```bash
pytest -q                          # 75 tests; needs no key and spends no tokens
python preflight.py                # 8 environment checks, none of them billed
python -m lab.evaluate_rules       # what the rules find, per case
streamlit run ui/app.py            # the interface
```

The dataset comes from `python -m lab.get_data`, which puts it at
`data/labels/` where the code looks by default. `LAB_DATA_DIR` is only needed
if you keep it somewhere else.
