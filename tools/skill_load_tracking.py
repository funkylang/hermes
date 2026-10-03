"""In-session skill load tracker for observability (/skills loaded).

Mirrors the skills_tool_dedup registry pattern: task_id-keyed, thread-safe, in-memory only.
Lifetime is the session — deliberately NOT reset on context compression (loaded skills
remain in the summarized conversation and stay "in context" until the session ends).

One event per unique (name, file_path) pair, first-seen order; repeat views increment
count instead of adding rows, so /skills loaded reads as "which skills, how often".
"""

import time
import threading

# task_id -> list of load events (dicts), one per unique (name, file_path)
_skill_load_tracker: dict = {}
_skill_load_tracker_lock = threading.Lock()
_MAX_EVENTS_PER_SESSION = 500


def record_skill_load(task_id, name, file_path=None, *, repeat=False, error=False):
    """Record a skill_view call. ``repeat`` = dedup stub (content served earlier); it only
    bumps the count on the existing event, never adds one."""
    if not task_id:
        return
    with _skill_load_tracker_lock:
        events = _skill_load_tracker.setdefault(str(task_id), [])
        for e in events:
            if e["name"] == str(name) and e["file_path"] == (file_path or None):
                e["count"] += 1
                e["repeat"] = e.get("repeat", False) or bool(repeat)
                e["ts"] = time.time()  # last access
                return
        event = {
            "name": str(name),
            "file_path": file_path or None,
            "first_ts": time.time(),
            "ts": time.time(),  # last access (updated on every subsequent view)
            "repeat": bool(repeat),  # True once any repeat view has been seen for this skill
            "error": bool(error),
            "count": 1,
        }
        if len(events) >= _MAX_EVENTS_PER_SESSION:
            events.pop(0)                      # bound memory; oldest drops first
        events.append(event)


def get_session_skill_loads(task_id):
    """All load events for a session, in first-seen order. [] when unknown/empty."""
    with _skill_load_tracker_lock:
        return list(_skill_load_tracker.get(str(task_id), ())) if task_id else []


def clear_session_skill_loads(task_id=None):
    """Drop the tracker entry (all sessions when task_id is None)."""
    with _skill_load_tracker_lock:
        if task_id is None:
            _skill_load_tracker.clear()
        else:
            _skill_load_tracker.pop(str(task_id), None)
