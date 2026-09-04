"""Tests for the console metric vocabulary and the metrics built on it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lab import opi

GOLDEN = "case_045"


def test_dictionary_has_109_unique_metrics():
    d = opi.metric_dictionary()
    assert len(d) == 109
    assert len(set(d)) == 109
    assert d["cam_clutch_cnt"]["display_name"] == "endoscope clutch count"


def test_loader_exposes_no_value_columns():
    forbidden = {
        "metric_value",
        "arm",
        "console",
        "metric_id",
        "created_at",
        "updated_at",
        "anno_card_id",
    }
    for entry in opi.metric_dictionary().values():
        assert not (forbidden & set(entry)), entry


def test_only_three_metrics_are_derivable():
    derivable = {m for m, e in opi.metric_dictionary().items() if e["derivable"]}
    assert derivable == {"duration", "duration_tool", "duration_armxtool"}
    assert opi.is_derivable("duration") is True
    assert opi.is_derivable("jerk_all") is False


from lab.data import Case, load_case
from lab.metrics import install_durations


def _case_from_rows(tools_rows) -> Case:
    tools = pd.DataFrame(
        tools_rows, columns=["part", "start_s", "end_s", "arm", "tool"]
    )
    tools["commercial"] = tools["tool"]
    tools["duration_s"] = tools["end_s"] - tools["start_s"]
    tasks = pd.DataFrame(
        [[1, 0.0, 100.0, "Suturing", 100.0]],
        columns=["part", "start_s", "end_s", "task", "duration_s"],
    )
    return Case("case_synthetic", tasks, tools)


def test_install_durations_clips_to_the_window():
    case = _case_from_rows([[1, -50.0, 50.0, "USM1", "needle driver"]])
    out = install_durations(case, part=1, start_s=0.0, end_s=100.0)
    assert out["duration_tool"]["needle driver"] == pytest.approx(50.0)
    assert out["duration_armxtool"]["USM1 needle driver"] == pytest.approx(50.0)


def test_install_durations_merges_overlapping_mounts_on_one_arm():
    case = _case_from_rows(
        [
            [1, 0.0, 60.0, "USM3", "needle driver"],
            [1, 30.0, 90.0, "USM3", "needle driver"],
        ]
    )
    out = install_durations(case, part=1, start_s=0.0, end_s=100.0)
    assert out["duration_armxtool"]["USM3 needle driver"] == pytest.approx(90.0)
    assert out["duration_tool"]["needle driver"] == pytest.approx(90.0)


def test_install_durations_sums_the_same_tool_across_arms():
    case = _case_from_rows(
        [
            [1, 0.0, 40.0, "USM1", "needle driver"],
            [1, 0.0, 40.0, "USM3", "needle driver"],
        ]
    )
    out = install_durations(case, part=1, start_s=0.0, end_s=100.0)
    assert out["duration_armxtool"]["USM1 needle driver"] == pytest.approx(40.0)
    assert out["duration_tool"]["needle driver"] == pytest.approx(80.0)


def test_install_durations_never_exceeds_the_window_per_arm():
    case = load_case(GOLDEN)
    seg = case.tasks[case.tasks["task"] == "Suturing"].iloc[0]
    out = install_durations(case, seg.part, seg.start_s, seg.end_s)
    window = seg.end_s - seg.start_s
    for key, seconds in out["duration_armxtool"].items():
        assert seconds <= window + 1e-6, key
    assert out["duration_armxtool"]["USM3 needle driver"] == pytest.approx(
        window, rel=1e-6
    )


from lab.metrics import get_metrics


def test_glossary_covers_every_step_field():
    result = get_metrics(GOLDEN, step="Suturing")
    fields = set(result["segments"][0])
    assert fields <= set(result["glossary"]), fields - set(result["glossary"])


def test_glossary_names_the_console_metric_for_duration():
    entry = get_metrics(GOLDEN, step="Suturing")["glossary"]["duration_s"]
    assert entry["source"] == "console"
    assert entry["console_name"] == "duration"
    assert "total time spent" in entry["definition"].lower()


def test_glossary_marks_lab_only_fields():
    entry = get_metrics(GOLDEN, step="Suturing")["glossary"]["swaps_per_min"]
    assert entry["source"] == "lab"


def test_unsupported_metric_returns_no_number():
    result = get_metrics(GOLDEN, metric="active_any_ratio_ssc")
    assert result["unavailable"] == "active_any_ratio_ssc"
    assert result["display_name"] == "console movement %"
    assert "telemetry" in result["reason"]
    assert not any(isinstance(v, (int, float)) for v in result.values())


def test_unknown_metric_says_so():
    result = get_metrics(GOLDEN, metric="not_a_metric")
    assert result["unknown_metric"] == "not_a_metric"


def test_derivable_metric_returns_computed_values():
    result = get_metrics(GOLDEN, step="Suturing", metric="duration_armxtool")
    assert result["metric"] == "duration_armxtool"
    assert result["segments"][0]["duration_armxtool"]["USM3 needle driver"] > 0


def test_no_lab_module_reads_the_dummy_table():
    lab_dir = Path(__file__).resolve().parent.parent / "lab"
    offenders = [
        path.name
        for path in lab_dir.rglob("*.py")
        if "combined_metrics" in path.read_text()
    ]
    assert offenders == [], offenders
