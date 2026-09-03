# Working on this repository

For instructors and anyone improving the lab. Participants do not need this.

## The one rule that matters

**`lab/` holds exercises. `solutions/` holds answers. They must not swap.**

Six functions across four files raise `NotImplementedError`, and that is the
product. It is easy to copy a reference solution over a skeleton while
testing, and easy to forget — so a test asserts all six still raise, and CI
runs it on every push.

```bash
pytest -q tests/test_supplied.py::test_participant_files_still_need_writing
```

If you need a working system to try something, **copy into a scratch checkout
rather than into `lab/`**. If you did overwrite one, `git checkout -- lab/`
puts it back.

The skeletons carry guidance in their **module docstrings** and nothing in
their bodies. That line is worth holding: guidance can be as long as it needs
to be, a body cannot exist at all. Watch the docstring length too — Lab 2's
reached 77 lines and 42% of the file before anyone noticed it had become a
wall of text.

## What lives where

| | |
|---|---|
| this repo, public | what participants clone: `lab/`, `solutions/`, `ui/`, `tests/`, `preflight.py` |
| `surgical-agent-lab-instructor`, private | client documents, the delivery plan, and the clip pipeline |

**Nothing identifying the client belongs in this repo.** That includes commit
messages. It has been breached once, and fixing it needed a history rewrite
and a force-push, which is only cheap before anyone has cloned.

The instructor pipeline is not here on purpose. It reads raw video and writes
the artefacts this repo already ships, so a participant browsing this
repository can run everything they can see.

## How to decide things

The parts of this lab that work were not designed, they were measured. Two
habits did most of it.

**Measure before choosing.** An "unexpected instrument for this step" rule was
an obvious candidate and was cut, because monopolar scissors are mounted
during 98% of *Uterine horn* segments — it would have flagged the norm. The
same test later showed `step_oscillation` fires on 81% of sessions, which is
why the participant guide points at it as the stretch exercise instead of
presenting it as a finding. `python -m lab.evaluate_rules` is the tool.

**Prove a test fails before you trust it.** Revert the fix, watch it go red,
put it back. Three tests written in one day looked fine and could not fail: one
asserted an f-string interpolates, one that `1.4` needs `:g` to print as
`"1.4"`, one set a Streamlit widget's state without the widget having a key,
so it silently tested the default case instead of the one it named. A test
that cannot fail is worse than no test, because it claims coverage that is not
there.

Two smaller ones. **Never quote a number you have not run** — the docstring
lesson in Lab 3 is a measured four-run table, not an assertion, and it is more
convincing for it. And **write the reason in the commit message**: there are
around 970 lines of them here, and they are where "why is it like this" is
actually answered.

## Before you push

```bash
pytest -q                # 90 tests, no key, no tokens
python preflight.py      # 10 checks, none billed
```

CI runs the same on 3.12 and 3.13. The floor is **3.12**, set by
`numpy 2.5.2`. That answer took two wrong ones first: `google-adk` says
`>=3.10`, `pandas` says `>=3.11`, and both were shipped as the floor before
anyone checked all 75 pins. **If you change a pin, re-derive the floor from
every pin, not from the one you changed.** CI failing on the lower matrix
version is the backstop.

## The recurring task

Clip URIs live **exactly 48 hours**. Refresh them on the morning of a session,
and remember it is three steps, not one: upload, push `lab/clips.json`, and
tell the room to `git pull`. `instructor/README.md` in the private repo has
the commands. Skipping either of the last two fails Lab 2 for everyone.
