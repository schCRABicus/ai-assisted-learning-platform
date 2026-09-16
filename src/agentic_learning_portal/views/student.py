"""Student endpoint of the Agentic Learning Portal.

Served at ``/student`` (via ``st.navigation`` in ``app.py``). Lists the
assignments assigned to the signed-in student, newest first, each in a bordered
card with its task count, status, and — once submitted — the graded score.

The card's button navigates to the take page (``student_take.py``), passing the
assignment id through session state (it survives ``st.switch_page``), exactly as
the admin assignments page reaches the editor. The label reflects the state the
attempt-flow rules put the student in: **▶️ Start** before the first attempt,
**↩️ Resume** while one is in progress, and **👁 View results** once they have
used every attempt they are entitled to.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from agentic_learning_portal.assignment import (
    active_attempt,
    attempt_score,
    can_start_attempt,
    student_attempts,
)
from agentic_learning_portal.auth import get_storage, require_roles
from agentic_learning_portal.storage import Assignment, Attempt, Storage

TAKE_PAGE = str(Path(__file__).parent / "student_take.py")


def _status(
    storage: Storage, assignment: Assignment, attempts: list[Attempt]
) -> tuple[str, str]:
    """Return the card's ``(status, score)`` line for an assignment.

    ``status`` is one of Not started / In progress / Submitted; ``score`` is the
    ``correct/graded`` tally of the newest graded attempt, or ``—``.
    """
    active = active_attempt(attempts)
    if active is not None:
        return "In progress", "—"
    if not attempts:
        return "Not started", "—"

    latest = attempts[-1]
    correct, graded, _ = attempt_score(storage, latest)
    return "Submitted", f"{correct}/{graded}" if graded else "—"


def _button_label(assignment: Assignment, attempts: list[Attempt]) -> str:
    """Return the call-to-action label matching the student's current state."""
    if active_attempt(attempts) is not None:
        return "↩️ Resume"
    if can_start_attempt(assignment, attempts) and not attempts:
        return "▶️ Start"
    if can_start_attempt(assignment, attempts):
        return "🔁 Start new attempt"
    return "👁 View results"


@require_roles()
def render_student_page() -> None:
    st.set_page_config(page_title="🧑🎓 Student", page_icon="🧑🎓")

    user = st.session_state["user"]
    st.title("🧑🎓 Student")
    st.caption(f"Signed in as **{user.username}**.")

    storage = get_storage()
    assignments = sorted(
        storage.list_assignments(assigned_to=user.id),
        key=lambda a: a.created_at,
        reverse=True,
    )

    if not assignments:
        st.info("No assignments have been assigned to you yet.")
        st.stop()

    for assignment in assignments:
        attempts = student_attempts(storage, assignment.id, user.id)
        status, score = _status(storage, assignment, attempts)
        active = active_attempt(attempts)
        can_start = can_start_attempt(assignment, attempts)

        with st.container(border=True):
            col_title, col_action = st.columns([5, 1], vertical_alignment="center")
            with col_title:
                st.markdown(f"### {assignment.title}")
                st.markdown(f"**📦 Tasks:** {assignment.tasks.size}")
                st.markdown(f"**🚦 Status:** {status} · **📊 Score:** {score}")
                if not can_start and active is None:
                    st.caption(
                        "You have used every attempt for this assignment. "
                        "Ask your teacher for another one."
                    )
            with col_action:
                if st.button(
                    _button_label(assignment, attempts),
                    key=f"open_assignment_{assignment.id}",
                    use_container_width=True,
                    help="Open this assignment.",
                ):
                    st.session_state["take_assignment_id"] = assignment.id
                    st.switch_page(TAKE_PAGE)


render_student_page()
