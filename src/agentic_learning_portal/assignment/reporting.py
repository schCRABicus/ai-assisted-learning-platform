"""Display helpers for rendering attempts in the admin views.

Both the assignments overview and the assignment editor show the same things
about an attempt — who made it, when, and how it scored — so the formatting lives
here rather than being copied into each page. Everything degrades to a readable
placeholder instead of raising: a missing user or a malformed timestamp should
never take down an admin page.
"""

from __future__ import annotations

from datetime import datetime

from agentic_learning_portal.assignment.attempts import attempt_score
from agentic_learning_portal.storage import Attempt, Storage

PLACEHOLDER = "—"


def format_timestamp(iso: str | None) -> str:
    """Render an ISO-8601 timestamp as ``%Y-%m-%d %H:%M``, or ``—`` if absent."""
    if not iso:
        return PLACEHOLDER
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso


def username(storage: Storage, user_id: int | None) -> str:
    """Resolve a user id to a username, degrading to ``user #<id>`` on failure."""
    if user_id is None:
        return PLACEHOLDER
    try:
        user = storage.get_user(user_id)
    except Exception:  # noqa: BLE001 - best-effort display helper
        return f"user #{user_id}"
    return user.username if user is not None else f"user #{user_id}"


def score_line(storage: Storage, attempt: Attempt) -> str:
    """Render an attempt's graded score as ``X/Y`` plus its average, or ``—``."""
    correct, graded, average = attempt_score(storage, attempt)
    if not graded:
        return PLACEHOLDER
    line = f"{correct}/{graded}"
    if average is not None:
        line += f" (avg {average:.2f})"
    return line
