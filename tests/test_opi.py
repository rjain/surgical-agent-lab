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
