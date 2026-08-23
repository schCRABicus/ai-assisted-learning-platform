"""Assignments overview endpoint of the Agentic Learning Portal.

Served at ``/assignments`` (via ``st.navigation`` in ``app.py``). The page
lists every assignment in the portal — no owner/student filter — newest first,
each in a bordered card with its task count, who it's assigned to, who created
it, and the metadata of its latest attempt (when it was last attempted, the
graded score, and whether it is still in progress). Purely read-only over
storage: no LLM calls, no threading, no session state. Task counts come from
each assignment's joined ``tasks.size``; task contents are fetched lazily
through the ``LazyTaskList`` adapter on first access (inside the expander).
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from agentic_learning_portal.auth import get_storage, require_roles
from agentic_learning_portal.storage import Attempt, Storage


def _format_timestamp(iso: str | None) -> str:
    """Render an ISO-8601 timestamp as ``%Y-%m-%d %H:%M``, or ``"—"`` if absent."""
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso


def _username(storage: Storage, user_id: int) -> str:
    """Resolve a user id to a username, degrading to ``user #<id>`` on failure."""
    try:
        user = storage.get_user(user_id)
    except Exception:  # noqa: BLE001 - best-effort display helper
        return f"user #{user_id}"
    return user.username if user is not None else f"user #{user_id}"


def _last_attempt(storage: Storage, assignment_id: int) -> Attempt | None:
    """Return the most recent attempt for an assignment, or ``None`` if none."""
    attempts = storage.list_attempts(assignment_id=assignment_id)
    return attempts[-1] if attempts else None


def _attempt_score(storage: Storage, attempt: Attempt) -> tuple[int, int, float | None]:
    """Summarize an attempt's results as ``(correct, graded, average_score)``."""
    results = storage.list_results(attempt_id=attempt.id)
    graded = [r for r in results if r.is_correct is not None]
    correct = sum(1 for r in graded if r.is_correct)
    scores = [r.score for r in results if r.score is not None]
    avg = sum(scores) / len(scores) if scores else None
    return correct, len(graded), avg

# Define the confirmation pop-up dialog
@st.dialog("Confirm Assignment Deletion", dismissible=True)
def delete_assignment_dialog(assignment_id: int) -> None:
  st.write("Are you sure you want to delete this assignment? This cannot be undone.")

  col1, col2 = st.columns(2)

  if col1.button("Yes, Delete", type="primary"):
    get_storage().delete_assignment(assignment_id)
    st.success("Item deleted successfully!")
    # st.session_state.show_dialog = False
    st.rerun()

  if col2.button("Cancel"):
    # st.session_state.show_dialog = False
    st.rerun()

def render_page() -> None:
    # Gate the page before any widget — mirrors views/admin.py.
    require_roles("admin", "teacher")

    st.set_page_config(page_title="🎓 Assignments", page_icon="🎓")

    st.title("🎓 Assignments")
    st.caption("All assignments in the portal, with their latest attempt results.")

    storage = get_storage()
    assignments = sorted(storage.list_assignments(), key=lambda a: a.created_at, reverse=True)

    if not assignments:
        st.info("No assignments yet. Create one from the Admin page.")
        st.stop()

    for assignment in assignments:
        attempt = _last_attempt(storage, assignment.id)
        if attempt is None:
            last_at, status, score = "Never", "Not attempted", "—"
        else:
            last_at = _format_timestamp(attempt.completed_at or attempt.started_at)
            status = "Completed" if attempt.status == "completed" else "In progress"
            correct, total, avg = _attempt_score(storage, attempt)
            score = f"{correct}/{total}" if total else "—"
            if avg is not None and total:
                score += f" (avg {avg:.2f})"

        with st.container(border=True):
            col_title, col_edit, col_delete = st.columns([1, 1, 1])
            with col_title:
                st.markdown(f"### {assignment.title}")
            with col_edit:
                if st.button("Edit", key=f"edit_{assignment.id}"):
                    st.session_state.assignment_edited = assignment.id
                    st.rerun()
            with col_delete:
                st.button("Delete", key=f"delete_{assignment.id}", icon=":material/delete:", on_click=lambda:delete_assignment_dialog(assignment.id))


            meta_row1 = st.columns(3)
            meta_row1[0].markdown(f"**📦 Tasks:** {assignment.tasks.size}")
            meta_row1[1].markdown(
                f"**👤 Assigned to:** {_username(storage, assignment.assigned_to) if assignment.assigned_to is not None else '—'}"
            )
            meta_row1[2].markdown(f"**✍️ Created by:** {_username(storage, assignment.created_by)}")
            meta_row2 = st.columns(3)
            meta_row2[0].markdown(f"**🕒 Last attempted:** {last_at}")
            meta_row2[1].markdown(f"**📊 Score:** {score}")
            meta_row2[2].markdown(f"**🚦 Status:** {status}")
            st.caption(f"id {assignment.id} · created {_format_timestamp(assignment.created_at)}")
            with st.expander(f"📝 {assignment.tasks.size} tasks"):
                if assignment.tasks:
                    for task in assignment.tasks:
                        st.markdown(f"- **{task.topic}** — {task.complexity}")
                else:
                    st.markdown("No tasks yet.")


render_page()
