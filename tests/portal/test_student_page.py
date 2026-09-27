"""Tests for the student assignments list (views/student.py).

The page is driven through the login form, as ``tests/admin/test_admin_page.py``
does, so the guard and the list are exercised end to end. Storage is the
in-memory backend from the ``portal_env`` fixture.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.auth import get_storage
from agentic_learning_portal.domains.math import grade_attempt

STUDENT_PAGE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "agentic_learning_portal"
    / "views"
    / "student.py"
)

TAKE_PAGE = str(STUDENT_PAGE.parent / "student_take.py")


def _login_as(at: AppTest, username: str, password: str) -> None:
    at.text_input[0].set_value(username)
    at.text_input[1].set_value(password)
    at.button[0].click().run()


def _student(username: str = "pupil"):
    return get_storage().create_user(username, "student", password="pw123")


def _assignment(
    title: str = "Algebra HW",
    *,
    assigned_to: int | None = None,
    with_task: bool = True,
    extra_attempts: int = 0,
):
    """Seed one assignment (with a task unless told otherwise) for a student."""
    storage = get_storage()
    admin = storage.get_user_by_username("boss")
    assignment = storage.create_assignment(
        title=title, created_by=admin.id, assigned_to=assigned_to
    )
    if with_task:
        task = storage.create_task(
            GeneratedTask(
                topic="Algebra",
                text="What is 2 + 2?",
                complexity="easy",
                correct_answer="4",
                solution="Add the numbers.",
            )
        )
        storage.add_task_to_assignment(assignment.id, task.id)
    for _ in range(extra_attempts):
        assignment = storage.grant_extra_attempt(assignment.id)
    return assignment


def _submit(assignment_id: int, student_id: int, answer: str = "4") -> None:
    """Take and submit an assignment for a student, leaving a graded attempt."""
    storage = get_storage()
    task = storage.list_assignment_tasks(assignment_id)[0]
    attempt = storage.start_attempt(assignment_id, student_id)
    storage.record_result(attempt.id, task.id, given_answer=answer)

    grade_attempt(storage, attempt.id)
    storage.complete_attempt(attempt.id)


def _card_label(at: AppTest, assignment_id: int) -> str | None:
    """Return the call-to-action label on ``assignment_id``'s card."""
    for button in at.button:
        if button.key == f"open_assignment_{assignment_id}":
            return button.label
    return None


def _marks(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown]


def test_student_page_requires_authentication(portal_env) -> None:
    """An anonymous visitor sees the login form, never the assignment list."""
    pupil = _student()
    _assignment("Secret HW", assigned_to=pupil.id)

    at = AppTest.from_file(str(STUDENT_PAGE), default_timeout=10)
    at.run()

    assert not at.exception
    assert any(t.label == "Username" for t in at.text_input)
    assert not at.title
    assert not any("Secret HW" in m for m in _marks(at))


def test_student_page_lists_only_own_assignments(portal_env) -> None:
    """A student sees their assignments, not another student's."""
    pupil = _student()
    other = _student("classmate")
    _assignment("Mine", assigned_to=pupil.id)
    _assignment("Theirs", assigned_to=other.id)
    _assignment("Unassigned")

    at = AppTest.from_file(str(STUDENT_PAGE), default_timeout=10)
    at.run()
    _login_as(at, "pupil", "pw123")

    assert not at.exception
    marks = _marks(at)
    assert any("### Mine" in m for m in marks)
    assert not any("Theirs" in m for m in marks)
    assert not any("Unassigned" in m for m in marks)


def test_student_page_shows_empty_state(portal_env) -> None:
    _student()

    at = AppTest.from_file(str(STUDENT_PAGE), default_timeout=10)
    at.run()
    _login_as(at, "pupil", "pw123")

    assert not at.exception
    assert any("No assignments have been assigned" in i.value for i in at.info)


def test_card_shows_task_count_and_not_started_status(portal_env) -> None:
    pupil = _student()
    _assignment("Algebra HW", assigned_to=pupil.id)

    at = AppTest.from_file(str(STUDENT_PAGE), default_timeout=10)
    at.run()
    _login_as(at, "pupil", "pw123")

    marks = _marks(at)
    assert any("**📦 Tasks:** 1" in m for m in marks)
    assert any("**🚦 Status:** Not started" in m and "**📊 Score:** —" in m for m in marks)


def test_button_offers_start_before_any_attempt(portal_env) -> None:
    pupil = _student()
    assignment = _assignment(assigned_to=pupil.id)

    at = AppTest.from_file(str(STUDENT_PAGE), default_timeout=10)
    at.run()
    _login_as(at, "pupil", "pw123")

    assert _card_label(at, assignment.id) == "▶️ Start"


def test_button_offers_resume_while_in_progress(portal_env) -> None:
    pupil = _student()
    assignment = _assignment(assigned_to=pupil.id)
    get_storage().start_attempt(assignment.id, pupil.id)

    at = AppTest.from_file(str(STUDENT_PAGE), default_timeout=10)
    at.run()
    _login_as(at, "pupil", "pw123")

    assert _card_label(at, assignment.id) == "↩️ Resume"
    assert any("**🚦 Status:** In progress" in m for m in _marks(at))


def test_button_shows_view_results_once_submitted(portal_env) -> None:
    pupil = _student()
    assignment = _assignment(assigned_to=pupil.id)
    _submit(assignment.id, pupil.id, answer="4")

    at = AppTest.from_file(str(STUDENT_PAGE), default_timeout=10)
    at.run()
    _login_as(at, "pupil", "pw123")

    assert _card_label(at, assignment.id) == "👁 View results"
    marks = _marks(at)
    assert any("**🚦 Status:** Submitted" in m and "**📊 Score:** 1/1" in m for m in marks)
    assert any("used every attempt" in c.value for c in at.caption)


def test_granted_retake_offers_a_new_attempt(portal_env) -> None:
    """After the admin grants one more attempt, the card offers a fresh one."""
    pupil = _student()
    assignment = _assignment(assigned_to=pupil.id)
    _submit(assignment.id, pupil.id, answer="4")
    get_storage().grant_extra_attempt(assignment.id)

    at = AppTest.from_file(str(STUDENT_PAGE), default_timeout=10)
    at.run()
    _login_as(at, "pupil", "pw123")

    assert _card_label(at, assignment.id) == "🔁 Start new attempt"
    assert not any("used every attempt" in c.value for c in at.caption)


def test_button_navigates_to_take_page_with_the_clicked_assignment(portal_env) -> None:
    """The card button stashes the assignment id and switches to the take page.

    Two assignments, so a closure-in-loop bug (every button opening the last
    assignment) would show up here.
    """
    pupil = _student()
    first = _assignment("Alpha", assigned_to=pupil.id)
    _assignment("Beta", assigned_to=pupil.id)

    with patch("streamlit.switch_page") as switch:
        at = AppTest.from_file(str(STUDENT_PAGE), default_timeout=10)
        at.run()
        _login_as(at, "pupil", "pw123")
        at.button(key=f"open_assignment_{first.id}").click().run()

    assert not at.exception
    assert at.session_state["take_assignment_id"] == first.id
    switch.assert_called_once_with(TAKE_PAGE)
