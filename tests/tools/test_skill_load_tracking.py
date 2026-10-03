"""Tests for tools.skill_load_tracking (in-session skill load observability)."""

import time

import pytest

from tools import skill_load_tracking as slt


@pytest.fixture(autouse=True)
def _clean_tracker():
    slt.clear_session_skill_loads(None)
    yield
    slt.clear_session_skill_loads(None)


def test_record_and_get_roundtrip():
    before = time.time()
    slt.record_skill_load("t1", "plan")
    events = slt.get_session_skill_loads("t1")
    assert len(events) == 1
    e = events[0]
    assert e["name"] == "plan"
    assert e["file_path"] is None
    assert e["count"] == 1
    assert before <= e["first_ts"] <= time.time()


def test_unknown_task_id_returns_empty():
    assert slt.get_session_skill_loads("nope") == []
    assert slt.get_session_skill_loads(None) == []
    assert slt.get_session_skill_loads("") == []


def test_no_task_id_is_noop():
    slt.record_skill_load(None, "plan")
    slt.record_skill_load("", "plan")
    assert slt.get_session_skill_loads(None) == []
    # must not have created an empty-string bucket either
    assert slt.get_session_skill_loads("t1") == []


def test_file_path_and_flags_recorded():
    slt.record_skill_load("t1", "plan", file_path="references/api.md", repeat=True)
    e = slt.get_session_skill_loads("t1")[0]
    assert e["file_path"] == "references/api.md"
    assert e["repeat"] is True


def test_consecutive_duplicates_collapse():
    slt.record_skill_load("t1", "plan")
    slt.record_skill_load("t1", "plan")
    slt.record_skill_load("t1", "plan")
    events = slt.get_session_skill_loads("t1")
    assert len(events) == 1
    assert events[0]["count"] == 3


def test_repeat_views_aggregate_regardless_of_order():
    # non-consecutive repeat views of the same skill roll into one event (first-seen slot)
    slt.record_skill_load("t1", "plan")
    slt.record_skill_load("t1", "text")
    slt.record_skill_load("t1", "plan", repeat=True)
    events = slt.get_session_skill_loads("t1")
    assert [e["name"] for e in events] == ["plan", "text"]
    assert events[0]["count"] == 2
    assert events[0]["repeat"] is True


def test_repeat_then_full_share_event_slot():
    # a repeat stub arriving right after a load of the same (name, file) collapses into it
    slt.record_skill_load("t1", "plan")
    slt.record_skill_load("t1", "plan", repeat=True)
    events = slt.get_session_skill_loads("t1")
    assert len(events) == 1
    assert events[0]["count"] == 2


def test_separate_sessions_isolated():
    slt.record_skill_load("t1", "plan")
    slt.record_skill_load("t2", "text")
    assert [e["name"] for e in slt.get_session_skill_loads("t1")] == ["plan"]
    assert [e["name"] for e in slt.get_session_skill_loads("t2")] == ["text"]


def test_clear_session_specific():
    slt.record_skill_load("t1", "plan")
    slt.record_skill_load("t2", "text")
    slt.clear_session_skill_loads("t1")
    assert slt.get_session_skill_loads("t1") == []
    assert len(slt.get_session_skill_loads("t2")) == 1
    slt.clear_session_skill_loads(None)
    assert slt.get_session_skill_loads("t2") == []


def test_capacity_bound():
    for i in range(slt._MAX_EVENTS_PER_SESSION + 50):
        # distinct names so nothing collapses
        slt.record_skill_load("t1", f"skill-{i}")
    events = slt.get_session_skill_loads("t1")
    assert len(events) == slt._MAX_EVENTS_PER_SESSION
    # oldest dropped: 550 inserted - 500 kept = first 50 evicted
    assert events[0]["name"] == f"skill-{(slt._MAX_EVENTS_PER_SESSION + 50) - slt._MAX_EVENTS_PER_SESSION}"


# --- handler wiring: _skill_view_with_bump records into the tracker ---

def test_skill_view_with_bump_records_load(monkeypatch):
    import tools.skills_tool as st

    fake_result = '{"success": true, "name": "plan", "content": "# plan"}'
    monkeypatch.setattr(st, "skill_view", lambda *a, **kw: fake_result)
    # neutralize usage bumps to keep the test hermetic
    import tools.skill_usage as su
    monkeypatch.setattr(su, "bump_view", lambda *a, **kw: None)
    monkeypatch.setattr(su, "bump_use", lambda *a, **kw: None)

    st._skill_view_with_bump({"name": "plan"}, task_id="t-x")
    events = slt.get_session_skill_loads("t-x")
    assert len(events) == 1
    assert events[0]["name"] == "plan"


def test_dedup_stub_records_repeat(monkeypatch):
    import tools.skills_tool as st

    # Seed the dedup cache so a repeat view returns the stub path.
    payload = {
        "success": True,
        "name": "plan",
        "_source_path": "/tmp/does-not-matter/SKILL.md",
    }
    slt.clear_session_skill_loads(None)
    # _record_skill_view stats the file; point at a real one (this test file).
    import os
    real = os.path.abspath(__file__)
    st._record_skill_view("t-y", "plan", None, {**payload, "_source_path": real})

    st._skill_view_with_bump({"name": "plan"}, task_id="t-y")
    events = slt.get_session_skill_loads("t-y")
    assert len(events) == 1
    assert events[0]["repeat"] is True


def test_failed_view_not_recorded(monkeypatch):
    import tools.skills_tool as st

    monkeypatch.setattr(
        st, "skill_view",
        lambda *a, **kw: '{"success": false, "error_message": "Skill not found"}')

    st._skill_view_with_bump({"name": "ghost"}, task_id="t-z")
    assert slt.get_session_skill_loads("t-z") == []
