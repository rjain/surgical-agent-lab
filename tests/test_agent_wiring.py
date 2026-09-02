"""Tests for Lab 3's agent wiring and the tool contract it exposes.

No key, no tokens, no network. Constructing an `Agent` and asking ADK for a
tool's declaration are both local operations — which means the thing the model
will actually be shown is inspectable before anyone spends a token on it.

That is the point. A tool's declaration is built from its signature and its
docstring, so a docstring is a contract with the model rather than a comment
for a human. These tests read the generated declarations and check the
promises the lab makes about them.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ADK announces its JSON-schema function declarations as experimental on every
# agent build. Nothing here can act on it; it is upstream noise.
warnings.filterwarnings("ignore", message=r".*JSON_SCHEMA_FOR_FUNC_DECL.*")


def declarations(tools):
    """What the model is shown for each tool, as ADK generates it.

    ADK 2.8 puts the parameters in ``parameters_json_schema``, not the
    ``parameters`` field an older example would reach for — that one is None.
    """
    from google.adk.tools import FunctionTool

    return {
        (d := FunctionTool(fn)._get_declaration()).name: d.model_dump(exclude_none=True)
        for fn in tools
    }


@pytest.fixture(scope="module")
def reference():
    from solutions.lab3_agent import TOOLS, build_agent

    return build_agent(), declarations(TOOLS)


def test_the_reference_agent_holds_exactly_the_three_tools(reference):
    agent, decls = reference
    assert set(decls) == {"get_metrics", "list_deviations", "analyze_clip"}
    assert len(agent.tools) == 3


def test_the_agent_uses_the_lab_model_not_a_hardcoded_one(reference):
    """A pinned model id here would ignore LAB_MODEL and break the fallback."""
    from lab import config

    agent, _ = reference
    assert agent.model == config.model()


def test_the_agent_has_an_instruction_that_forbids_inventing_numbers(reference):
    agent, _ = reference
    instruction = agent.instruction.lower()
    assert "tool" in instruction
    assert any(word in instruction for word in ("never", "do not", "don't"))


@pytest.mark.parametrize("name", ["get_metrics", "list_deviations", "analyze_clip"])
def test_every_tool_declaration_reaches_the_model_complete(reference, name):
    """An empty description is a tool the model will pick at random."""
    _, decls = reference
    declaration = decls[name]
    assert declaration.get("description", "").strip(), f"{name} has no description"
    assert "parameters_json_schema" in declaration


def test_no_tool_schema_contains_a_type_the_api_rejects(reference):
    """`tuple` renders as `prefixItems`, which the Gemini API refuses outright
    — before any network call, with a message that never mentions tuples.

    Walks schema *keys*, not the serialised text: the word can legitimately
    appear inside a description that warns about it.
    """
    _, decls = reference

    def keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from keys(item)

    for name, declaration in decls.items():
        found = set(keys(declaration["parameters_json_schema"]))
        assert "prefixItems" not in found, f"{name} exposes a tuple"


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("get_metrics", ["case_id", "step"]),
        ("list_deviations", ["case_id"]),
        ("analyze_clip", ["case_id", "part", "t_start", "t_end"]),
    ],
)
def test_every_parameter_is_explained_in_the_docstring(reference, name, params):
    """The docstring is the contract. A parameter the model is asked for but
    never told about is the whole reason Lab 3's first agent misbehaves."""
    _, decls = reference
    declaration = decls[name]
    schema = declaration["parameters_json_schema"]
    assert set(schema["properties"]) == set(params)
    for param in params:
        assert param in declaration["description"], (
            f"{name} takes {param} but its docstring never mentions it"
        )


def test_analyze_clip_asks_for_a_part_alongside_its_offsets(reference):
    """Time restarts at zero in each video part, so an offset without a part
    is meaningless. The signature has to force the pairing."""
    _, decls = reference
    schema = decls["analyze_clip"]["parameters_json_schema"]
    assert "part" in schema["properties"]
    assert schema["properties"]["part"]["type"] == "integer"


def test_the_skeletons_get_metrics_docstring_is_still_deliberately_vague():
    """Load-bearing vagueness.

    Lab 3's teaching device is that participants run the agent with this
    docstring, watch it choose badly, then rewrite it. A well-meaning tidy-up
    — by a person or an agent — deletes the exercise, so this pins it.
    """
    from lab.lab3_agent import get_metrics

    docstring = (get_metrics.__doc__ or "").strip()
    assert docstring, "the vague docstring is the lesson; it must still be there"

    # The vagueness is in the content, not the structure. The skeleton keeps an
    # Args: block so participants can see the shape they are meant to fill —
    # what it does not say is what the tool is *for*, or what a step is.
    assert "the id" in docstring, (
        "lab/lab3_agent.py's get_metrics docstring has been filled in — that is "
        "the participant's exercise, and the written-out version already lives "
        "in solutions/lab3_agent.py."
    )
    assert len(docstring) < 200, "too detailed to still be the exercise"
    for giveaway in ("corpus", "median", "how long", "measured"):
        assert giveaway not in docstring.lower(), (
            f"{giveaway!r} tells the model what the tool is for — that is the "
            "answer, not the exercise"
        )


def test_the_reference_get_metrics_docstring_is_the_written_out_one():
    """The other half of the same lesson: solutions/ shows the fixed form."""
    from solutions.lab3_agent import get_metrics

    docstring = get_metrics.__doc__ or ""
    assert "Args:" in docstring
    assert "case_id" in docstring and "step" in docstring
