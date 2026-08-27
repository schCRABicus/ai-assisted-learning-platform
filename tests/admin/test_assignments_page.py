"""Tests for the assignments overview endpoint (views/admin/01_assignments.py)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from streamlit.testing.v1 import AppTest

from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.auth import get_storage
from unittest.mock import patch

ASSIGNMENTS_PAGE = Path(__file__).resolve().parents[2] / "src" / "agentic_learning_portal" / "views" / "admin" / "01_assignments.py"


def _login_as(at: AppTest, username: str, password: str) -> None:
    at.text_input[0].set_value(username)
    at.text_input[1].set_value(password)
    at.button[0].click().run()


def _login(at: AppTest) -> None:
    _login_as(at, "boss", "hunter2")


def _seed_assignment(*, title: str = "Algebra HW", with_attempt: bool = True):
    """Seed one assignment (+ one task, + optionally a completed attempt)."""
    storage = get_storage()
    admin = storage.get_user_by_username("boss")
    assignment = storage.create_assignment(title=title, created_by=admin.id)
    task = storage.create_task(
        GeneratedTask(topic="Algebra", text="What is 2 + 2?", complexity="easy",
                      correct_answer="4", solution="Add the numbers.")
    )
    storage.add_task_to_assignment(assignment.id, task.id)
    if with_attempt:
        attempt = storage.start_attempt(assignment.id, admin.id)
        storage.record_result(attempt.id, task.id, given_answer="4", expected_answer="4",
                              is_correct=True, score=1.0)
        attempt = storage.complete_attempt(attempt.id)
        return assignment, task, attempt
    return assignment, task, None


def test_assignments_page_requires_authentication(portal_env) -> None:
    """An anonymous visitor sees the login form, never the assignments list."""
    _seed_assignment(title="Secret HW")

    at = AppTest.from_file(str(ASSIGNMENTS_PAGE), default_timeout=10)
    at.run()

    assert not at.exception
    # The page is auth-gated: the login form renders and the page stops before
    # any assignment content (no title, no assignment cards).
    assert any(t.label == "Username" for t in at.text_input)
    assert any(t.label == "Password" for t in at.text_input)
    assert not at.title
    assert not any("Secret HW" in m.value for m in at.markdown)


def test_assignments_page_denies_non_admin_roles(portal_env) -> None:
    """A signed-in user without the admin/teacher roles is denied the page."""
    _seed_assignment(title="Secret HW")
    get_storage().create_user("pupil", "student", password="pw123")

    at = AppTest.from_file(str(ASSIGNMENTS_PAGE), default_timeout=10)
    at.run()
    _login_as(at, "pupil", "pw123")

    assert not at.exception
    assert any("Access denied" in e.value for e in at.error)
    assert not at.title
    assert not any("Secret HW" in m.value for m in at.markdown)


def test_admin_can_access_assignments_page(portal_env) -> None:
    """The env-seeded admin (boss) sees the assignments list."""
    _seed_assignment(title="Algebra HW")

    at = AppTest.from_file(str(ASSIGNMENTS_PAGE), default_timeout=10)
    at.run()
    _login(at)

    assert not at.exception
    assert at.title[0].value == "🎓 Assignments"
    assert any("### Algebra HW" in m.value for m in at.markdown)


def test_teacher_can_access_assignments_page(portal_env) -> None:
    """The page also admits the teacher role (require_roles admin|teacher)."""
    _seed_assignment(title="Geometry HW")
    get_storage().create_user("tess", "teacher", password="pw123")

    at = AppTest.from_file(str(ASSIGNMENTS_PAGE), default_timeout=10)
    at.run()
    _login_as(at, "tess", "pw123")

    assert not at.exception
    assert at.title[0].value == "🎓 Assignments"
    assert any("### Geometry HW" in m.value for m in at.markdown)


def test_assignments_page_shows_empty_state(portal_env) -> None:
    """The empty portal shows the 'No assignments yet' info callout."""
    at = AppTest.from_file(str(ASSIGNMENTS_PAGE), default_timeout=10)
    at.run()
    _login(at)

    assert not at.exception
    assert at.title[0].value == "🎓 Assignments"
    assert any("No assignments yet" in info.value for info in at.info)


def test_assignments_page_lists_assignment_with_attempt_metadata(portal_env) -> None:
    """An assignment with a completed attempt shows its attempt metadata."""
    assignment, task, attempt = _seed_assignment()

    at = AppTest.from_file(str(ASSIGNMENTS_PAGE), default_timeout=10)
    at.run()
    _login(at)

    assert not at.exception
    marks = [m.value for m in at.markdown]
    assert any("### Algebra HW" in m for m in marks)
    assert any("**📦 Tasks:** 1" in m for m in marks)
    assert any("**👤 Assigned to:** —" in m for m in marks)
    assert any("**✍️ Created by:** boss" in m for m in marks)
    expected_last = datetime.fromisoformat(attempt.completed_at).strftime("%Y-%m-%d %H:%M")
    assert any(f"**🕒 Last attempted:** {expected_last}" in m for m in marks)
    assert any("**📊 Score:** 1/1" in m for m in marks)
    assert any("**🚦 Status:** Completed" in m for m in marks)

    assert any("📝 1 tasks" in e.label for e in at.expander)


def test_assignments_page_without_attempts_shows_placeholders(portal_env) -> None:
    """An assignment with no attempts shows the placeholder metadata."""
    _seed_assignment(title="Geometry Basics", with_attempt=False)

    at = AppTest.from_file(str(ASSIGNMENTS_PAGE), default_timeout=10)
    at.run()
    _login(at)

    assert not at.exception
    marks = [m.value for m in at.markdown]
    assert any("### Geometry Basics" in m for m in marks)
    assert any("**📦 Tasks:** 1" in m for m in marks)
    assert any("**🕒 Last attempted:** Never" in m for m in marks)
    assert any("**📊 Score:** —" in m for m in marks)
    assert any("**🚦 Status:** Not attempted" in m for m in marks)


def test_each_assignment_card_has_a_delete_button(portal_env) -> None:
    """Every assignment card renders a Delete button keyed by its id."""
    a1, _, _ = _seed_assignment(title="Alpha")
    a2, _, _ = _seed_assignment(title="Beta")

    at = AppTest.from_file(str(ASSIGNMENTS_PAGE), default_timeout=10)
    at.run()
    _login(at)

    assert not at.exception
    labels = {b.key: b.label for b in at.button}
    assert labels.get(f"delete_{a1.id}") == "Delete"
    assert labels.get(f"delete_{a2.id}") == "Delete"


def test_delete_button_opens_dialog_for_the_clicked_assignment(portal_env) -> None:
    """Each Delete button opens the dialog for its own assignment.

    Regression test for the closure-in-loop bug where every Delete button
    opened the dialog for the last assignment in the list.
    """
    a1, _, _ = _seed_assignment(title="Alpha")
    a2, _, _ = _seed_assignment(title="Beta")

    target = (
        "agentic_learning_portal.views.components.delete_assignment_dialog"
        ".delete_assignment_dialog"
    )
    with patch(target) as dialog:
        at = AppTest.from_file(str(ASSIGNMENTS_PAGE), default_timeout=10)
        at.run()
        _login(at)
        at.button(key=f"delete_{a1.id}").click().run()

    assert not at.exception
    dialog.assert_called_once_with(a1.id)


def test_edit_button_navigates_to_assignment_editor(portal_env) -> None:
    """Each Edit button navigates to the editor for its own assignment.

    The clicked assignment's id is passed through session state (it survives
    ``st.switch_page``) and the editor page path is handed to ``switch_page``.
    Regression guard for the closure-in-loop bug that previously hit Delete:
    every Edit button must target its own assignment's id.
    """
    a1, _, _ = _seed_assignment(title="Alpha")
    a2, _, _ = _seed_assignment(title="Beta")

    editor_page = str(ASSIGNMENTS_PAGE.parent / "02_assignment_editor.py")
    with patch("streamlit.switch_page") as switch:
        at = AppTest.from_file(str(ASSIGNMENTS_PAGE), default_timeout=10)
        at.run()
        _login(at)
        at.button(key=f"edit_{a2.id}").click().run()

    assert not at.exception
    assert at.session_state["edit_assignment_id"] == a2.id
    switch.assert_called_once_with(editor_page)


def _confirm_delete(at: AppTest, assignment_id: int) -> None:
    """Open the delete dialog for ``assignment_id`` and confirm deletion.

    AppTest only renders a ``st.dialog`` body on runs where the opening widget
    is active, so the delete button is re-clicked on the same run as the
    dialog's "Yes, Delete" button for the confirm click to be processed.
    """
    at.button(key=f"delete_{assignment_id}").click().run()
    yes = [b for b in at.button if b.label == "Yes, Delete"]
    assert yes, "expected the confirmation dialog's Yes, Delete button"
    at.button(key=f"delete_{assignment_id}").click()
    yes[0].click()
    at.run()


def test_confirm_delete_removes_only_the_clicked_assignment(portal_env) -> None:
    """Confirming the dialog deletes the clicked assignment, never the others."""
    a1, _, _ = _seed_assignment(title="Doomed")
    a2, _, _ = _seed_assignment(title="Survivor")

    at = AppTest.from_file(str(ASSIGNMENTS_PAGE), default_timeout=10)
    at.run()
    _login(at)

    _confirm_delete(at, a1.id)

    assert not at.exception
    assert get_storage().get_assignment(a1.id) is None
    assert get_storage().get_assignment(a2.id) is not None
    marks = [m.value for m in at.markdown]
    assert not any("### Doomed" in m for m in marks)
    assert any("### Survivor" in m for m in marks)


def _cancel_delete(at: AppTest, assignment_id: int) -> None:
    """Open the delete dialog for ``assignment_id`` and dismiss it via Cancel."""
    at.button(key=f"delete_{assignment_id}").click().run()
    cancel = [b for b in at.button if b.label == "Cancel"]
    assert cancel, "expected the confirmation dialog's Cancel button"
    at.button(key=f"delete_{assignment_id}").click()
    cancel[0].click()
    at.run()


def test_cancel_delete_keeps_the_assignment(portal_env) -> None:
    """Dismissing the dialog with Cancel leaves the assignment untouched."""
    a1, _, _ = _seed_assignment(title="Stays")

    at = AppTest.from_file(str(ASSIGNMENTS_PAGE), default_timeout=10)
    at.run()
    _login(at)

    _cancel_delete(at, a1.id)

    assert not at.exception
    assert get_storage().get_assignment(a1.id) is not None
    assert any("### Stays" in m.value for m in at.markdown)


def test_delete_last_assignment_shows_empty_state(portal_env) -> None:
    """Deleting the only assignment drops the page to the empty state."""
    a1, _, _ = _seed_assignment(title="Sole")

    at = AppTest.from_file(str(ASSIGNMENTS_PAGE), default_timeout=10)
    at.run()
    _login(at)

    _confirm_delete(at, a1.id)

    assert not at.exception
    assert get_storage().get_assignment(a1.id) is None
    assert any("No assignments yet" in info.value for info in at.info)
