"""Assignments overview endpoint of the Agentic Learning Portal.

Served at ``/assignments`` (via ``st.navigation`` in ``app.py``). The page
lists every assignment in the portal — no owner/student filter — newest first,
each in a bordered card with its task count, who it's assigned to, who created
it, and the metadata of its latest attempt (when it was last attempted, the
graded score, and whether it is still in progress). No LLM calls, no threading.
The **Edit** button on a card navigates to the assignment editor page
(``02_assignment_editor.py``), passing the assignment id through session state;
**Delete** opens a confirmation dialog. Task counts come from each assignment's
joined ``tasks.size``; task contents are fetched lazily through the
``LazyTaskList`` adapter on first access (inside the expander).

Above the list sits the **📥 New results** panel: this is how the admin is told a
student submitted something. It shows every graded attempt whose results haven't
been looked at yet (``results_seen`` in storage), with a **👁 Mark as seen**
button to clear it and **✏️ Open** to jump to the editor for the full breakdown.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from agentic_learning_portal.assignment import (
    COMPLETED,
    format_timestamp,
    score_line,
    username,
)
from agentic_learning_portal.auth import get_storage, require_roles
from agentic_learning_portal.storage import Attempt, Storage
from agentic_learning_portal.views.components.delete_assignment_dialog import (
    delete_assignment_dialog,
)

EDITOR_PAGE = str(Path(__file__).parent / "02_assignment_editor.py")


def _last_attempt(storage: Storage, assignment_id: int) -> Attempt | None:
    """Return the most recent attempt for an assignment, or ``None`` if none."""
    attempts = storage.list_attempts(assignment_id=assignment_id)
    return attempts[-1] if attempts else None


def _open_delete_confirmation(aid: int) -> None:
    """Open the delete confirmation dialog for ``aid``."""
    delete_assignment_dialog(aid)


def _render_new_results(storage: Storage) -> None:
    """Render the panel of graded attempts the admin hasn't looked at yet.

    This is the portal's only submission notification: a student submitting an
    assignment leaves ``results_seen`` false, so it shows up here until the admin
    marks it seen. Nothing renders when there is nothing new.
    """
    fresh = [
        a
        for a in reversed(storage.list_attempts(results_seen=False))
        if a.status == COMPLETED
    ]
    if not fresh:
        return

    st.subheader(f"📥 New results ({len(fresh)})")
    for attempt in fresh:
        assignment = storage.get_assignment(attempt.assignment_id)
        title = (
            assignment.title
            if assignment is not None
            else f"assignment #{attempt.assignment_id}"
        )
        with st.container(border=True):
            col_info, col_seen, col_open = st.columns(
                [4, 1, 1], vertical_alignment="center"
            )
            with col_info:
                st.markdown(f"**{title}** — {username(storage, attempt.student_id)}")
                st.caption(
                    f"Score {score_line(storage, attempt)}"
                    f" · submitted {format_timestamp(attempt.completed_at)}"
                )
            with col_seen:
                if st.button(
                    "👁 Mark as seen",
                    key=f"seen_{attempt.id}",
                    use_container_width=True,
                ):
                    get_storage().mark_results_seen(attempt.id)
                    st.rerun()
            with col_open:
                if st.button(
                    "✏️ Open",
                    key=f"open_{attempt.id}",
                    use_container_width=True,
                ):
                    st.session_state.edit_assignment_id = attempt.assignment_id
                    st.switch_page(EDITOR_PAGE)
    st.markdown("---")


@require_roles("admin", "teacher")
def render_assignments_list_page() -> None:
    """
    Renders assignments list page with a table of assignments.
    :return:
    """
    st.set_page_config(page_title="🎓 Assignments", page_icon="🎓")

    st.title("🎓 Assignments")
    st.caption("All assignments in the portal, with their latest attempt results.")

    storage = get_storage()
    _render_new_results(storage)

    assignments = sorted(storage.list_assignments(), key=lambda a: a.created_at, reverse=True)

    if not assignments:
        st.info("No assignments yet. Create one from the Admin page.")
        st.stop()

    for assignment in assignments:
        attempt = _last_attempt(storage, assignment.id)
        if attempt is None:
            last_at, status, score = "Never", "Not attempted", "—"
        else:
            last_at = format_timestamp(attempt.completed_at or attempt.started_at)
            status = "Completed" if attempt.status == COMPLETED else "In progress"
            score = score_line(storage, attempt)

        with st.container(border=True):
            col_title, col_edit, col_delete = st.columns([1, 1, 1])
            with col_title:
                st.markdown(f"### {assignment.title}")
            with col_edit:
                if st.button("Edit", key=f"edit_{assignment.id}"):
                    # Navigate to the assignment editor, passing the id through
                    # session state (it survives the st.switch_page).
                    st.session_state.edit_assignment_id = assignment.id
                    st.switch_page(EDITOR_PAGE)
            with col_delete:
                st.button("Delete", key=f"delete_{assignment.id}", icon=":material/delete:",
                          on_click=lambda aid=assignment.id: _open_delete_confirmation(aid))


            meta_row1 = st.columns(3)
            meta_row1[0].markdown(f"**📦 Tasks:** {assignment.tasks.size}")
            meta_row1[1].markdown(
                f"**👤 Assigned to:** {username(storage, assignment.assigned_to)}"
            )
            meta_row1[2].markdown(f"**✍️ Created by:** {username(storage, assignment.created_by)}")
            meta_row2 = st.columns(3)
            meta_row2[0].markdown(f"**🕒 Last attempted:** {last_at}")
            meta_row2[1].markdown(f"**📊 Score:** {score}")
            meta_row2[2].markdown(f"**🚦 Status:** {status}")
            st.caption(f"id {assignment.id} · created {format_timestamp(assignment.created_at)}")
            with st.expander(f"📝 {assignment.tasks.size} tasks"):
                if assignment.tasks:
                    for task in assignment.tasks:
                        st.markdown(f"- **{task.topic}** — {task.complexity}")
                else:
                    st.markdown("No tasks yet.")


render_assignments_list_page()
