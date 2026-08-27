"""Tests for the assignment editor endpoint (views/admin/02_assignment_editor.py).

The editor renders a carousel over an assignment's tasks — each slide shows the
problem text, correct answer, and solution — with per-slide **Edit** and
**Remove** buttons that open the shared dialogs, plus a **➕ Add task** button
that opens the task-creation dialog. It is reached from the assignments page via
``st.switch_page``, which passes the assignment id through session state
(``edit_assignment_id``), so the tests pre-seed that key before running the page.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from streamlit.testing.v1 import AppTest

from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.auth import get_storage
from agentic_learning_portal.domains.math import MathProblemGenerator

EDITOR_PAGE = (
    Path(__file__).resolve().parents[2]
    / "src" / "agentic_learning_portal" / "views" / "admin" / "02_assignment_editor.py"
)
ASSIGNMENTS_PAGE = (
    Path(__file__).resolve().parents[2]
    / "src" / "agentic_learning_portal" / "views" / "admin" / "01_assignments.py"
)


def _login_as(at: AppTest, username: str, password: str) -> None:
    at.text_input[0].set_value(username)
    at.text_input[1].set_value(password)
    at.button[0].click().run()


def _login(at: AppTest) -> None:
    _login_as(at, "boss", "hunter2")


def _task(topic: str, text: str, answer: str = "4", solution: str = "Work it out step by step.") -> GeneratedTask:
    return GeneratedTask(
        topic=topic,
        text=text,
        complexity="easy",
        correct_answer=answer,
        solution=solution,
    )


def _seed_assignment(*, title: str = "Algebra HW", n_tasks: int = 1):
    """Seed an assignment with ``n_tasks`` tasks; returns ``(assignment, tasks)``."""
    storage = get_storage()
    admin = storage.get_user_by_username("boss")
    assignment = storage.create_assignment(title=title, created_by=admin.id)
    tasks = []
    for i in range(n_tasks):
        task = storage.create_task(
            _task(f"Topic {i + 1}", f"Problem text {i + 1}", answer=f"Answer {i + 1}", solution=f"Solution {i + 1}")
        )
        storage.add_task_to_assignment(assignment.id, task.id)
        tasks.append(task)
    return assignment, tasks


def _open_editor(assignment_id: int, *, timeout: int = 10) -> AppTest:
    """Launch the editor page pre-seeded with ``edit_assignment_id`` and sign in."""
    at = AppTest.from_file(str(EDITOR_PAGE), default_timeout=timeout)
    at.session_state["edit_assignment_id"] = assignment_id
    at.run()
    _login(at)
    return at


def _find_button(at: AppTest, label: str) -> AppTest | None:
    """Return the first button whose label contains ``label`` (case-insensitive)."""
    needle = label.lower()
    for button in at.button:
        if needle in button.label.lower():
            return button
    return None


def _find_text_input(at: AppTest, label: str) -> AppTest | None:
    """Return the first text input whose label contains ``label``."""
    needle = label.lower()
    for widget in at.text_input:
        if needle in widget.label.lower():
            return widget
    return None


def test_editor_page_requires_authentication(portal_env) -> None:
    """An anonymous visitor sees the login form, never the editor content."""
    _seed_assignment()

    at = AppTest.from_file(str(EDITOR_PAGE), default_timeout=10)
    at.session_state["edit_assignment_id"] = 1
    at.run()

    assert not at.exception
    assert any(t.label == "Username" for t in at.text_input)
    assert any(t.label == "Password" for t in at.text_input)
    assert not at.title


def test_editor_page_denies_non_admin_roles(portal_env) -> None:
    """A signed-in user without the admin/teacher roles is denied the page."""
    assignment, _ = _seed_assignment()
    get_storage().create_user("pupil", "student", password="pw123")

    at = AppTest.from_file(str(EDITOR_PAGE), default_timeout=10)
    at.session_state["edit_assignment_id"] = assignment.id
    at.run()
    _login_as(at, "pupil", "pw123")

    assert not at.exception
    assert any("Access denied" in e.value for e in at.error)
    assert not at.title


def test_admin_can_access_editor_page(portal_env) -> None:
    """The env-seeded admin (boss) can open the editor for an assignment."""
    assignment, _ = _seed_assignment(title="Algebra HW")

    at = _open_editor(assignment.id)

    assert not at.exception
    assert at.title[0].value == "✏️ Assignment Editor"
    assert any("### Algebra HW" in m.value for m in at.markdown)


def test_teacher_can_access_editor_page(portal_env) -> None:
    """The page also admits the teacher role (require_roles admin|teacher)."""
    assignment, _ = _seed_assignment(title="Geometry HW")
    get_storage().create_user("tess", "teacher", password="pw123")

    at = AppTest.from_file(str(EDITOR_PAGE), default_timeout=10)
    at.session_state["edit_assignment_id"] = assignment.id
    at.run()
    _login_as(at, "tess", "pw123")

    assert not at.exception
    assert at.title[0].value == "✏️ Assignment Editor"
    assert any("### Geometry HW" in m.value for m in at.markdown)


def test_editor_page_without_selected_assignment(portal_env) -> None:
    """No ``edit_assignment_id`` shows an info callout with a way back."""
    at = AppTest.from_file(str(EDITOR_PAGE), default_timeout=10)
    at.run()
    _login(at)

    assert not at.exception
    assert any("No assignment selected" in info.value for info in at.info)
    assert any(b.key == "editor_back_home" for b in at.button)


def test_editor_page_missing_assignment(portal_env) -> None:
    """A selected assignment that no longer exists shows an error."""
    assignment, _ = _seed_assignment()
    get_storage().delete_assignment(assignment.id)

    at = _open_editor(assignment.id)

    assert not at.exception
    assert any(f"Assignment #{assignment.id} not found" in e.value for e in at.error)
    assert any(b.key == "editor_back_home" for b in at.button)


def test_editor_carousel_shows_task_problem_answer_solution(portal_env) -> None:
    """Each slide renders the task's problem, correct answer, and solution."""
    assignment, tasks = _seed_assignment(n_tasks=2)

    at = _open_editor(assignment.id)

    assert not at.exception
    # Slide 1 is the first task.
    assert at.session_state["editor_current_index"] == 0
    marks = [m.value for m in at.markdown]
    assert any("Task 1 of 2" in m for m in marks)
    assert any("Problem text 1" in m for m in marks)
    assert any("Answer 1" in m for m in marks)
    assert any("Solution 1" in m for m in marks)
    assert any("**📚 Topic:** Topic 1" in m for m in marks)
    assert not any("Problem text 2" in m for m in marks)


def test_editor_carousel_next_and_prev(portal_env) -> None:
    """▶ moves to the next task and ◀ back, disabling at the ends."""
    assignment, tasks = _seed_assignment(n_tasks=2)

    at = _open_editor(assignment.id)

    def _button_keys(at: AppTest) -> set[str]:
        return {b.key for b in at.button}

    # On the first slide only ▶ (next) is available.
    assert "editor_prev" not in _button_keys(at)
    assert "editor_next" in _button_keys(at)

    # Next -> slide 2 shows task 2.
    at.button(key="editor_next").click().run()
    assert at.session_state["editor_current_index"] == 1
    marks = [m.value for m in at.markdown]
    assert any("Task 2 of 2" in m for m in marks)
    assert any("Problem text 2" in m for m in marks)
    assert not any("Problem text 1" in m for m in marks)
    # On the last slide ▶ is gone and ◀ is back.
    assert "editor_prev" in _button_keys(at)
    assert "editor_next" not in _button_keys(at)

    # Previous -> back to slide 1.
    at.button(key="editor_prev").click().run()
    assert at.session_state["editor_current_index"] == 0
    marks = [m.value for m in at.markdown]
    assert any("Problem text 1" in m for m in marks)
    assert not any("Problem text 2" in m for m in marks)


def test_edit_button_opens_dialog_prepopulated(portal_env) -> None:
    """The per-slide Edit button opens the dialog with the task's fields filled in."""
    assignment, tasks = _seed_assignment(n_tasks=1)
    task = tasks[0]

    at = _open_editor(assignment.id)
    at.button(key=f"edit_task_{task.id}").click().run()

    assert not at.exception
    assert at.session_state["edit_task_open"] == task.id
    assert at.text_input(key="edit_task_topic").value == task.topic
    assert at.text_input(key="edit_task_answer").value == str(task.correct_answer)
    assert at.text_area(key="edit_task_text").value == task.text
    assert at.text_area(key="edit_task_solution").value == task.solution


def test_edit_save_persists_changes_to_storage(portal_env) -> None:
    """Saving the dialog updates the task in storage and closes the dialog."""
    assignment, tasks = _seed_assignment(n_tasks=1)
    task = tasks[0]

    at = _open_editor(assignment.id)
    at.button(key=f"edit_task_{task.id}").click().run()
    at.text_input(key="edit_task_topic").set_value("Fractions")
    at.text_area(key="edit_task_text").set_value("What is 3/4 of 20?")
    at.button(key="edit_task_save").click().run()

    assert not at.exception
    updated = get_storage().get_task(task.id)
    assert updated.topic == "Fractions"
    assert updated.text == "What is 3/4 of 20?"
    # The dialog closes itself after saving.
    assert "edit_task_open" not in at.session_state


def test_edit_cancel_leaves_task_unchanged(portal_env) -> None:
    """Cancel closes the dialog without touching the task."""
    assignment, tasks = _seed_assignment(n_tasks=1)
    task = tasks[0]

    at = _open_editor(assignment.id)
    at.button(key=f"edit_task_{task.id}").click().run()
    at.text_input(key="edit_task_topic").set_value("Changed")
    at.button(key="edit_task_cancel").click().run()

    assert not at.exception
    assert get_storage().get_task(task.id).topic == task.topic
    assert "edit_task_open" not in at.session_state


def test_back_button_returns_to_assignments(portal_env) -> None:
    """The back button navigates to the assignments page."""
    assignment, _ = _seed_assignment()

    at = AppTest.from_file(str(EDITOR_PAGE), default_timeout=10)
    at.session_state["edit_assignment_id"] = assignment.id
    with patch("streamlit.switch_page") as switch:
        at.run()
        _login(at)
        at.button(key="editor_back").click().run()

    assert not at.exception
    switch.assert_called_once_with(str(ASSIGNMENTS_PAGE))


def test_switching_assignment_resets_carousel_and_dialog(portal_env) -> None:
    """Editing a different assignment resets the slide index and closes any dialog."""
    a1, tasks1 = _seed_assignment(title="Alpha", n_tasks=2)
    a2, _ = _seed_assignment(title="Beta", n_tasks=1)

    at = _open_editor(a1.id)

    # Move to slide 2 and open its edit dialog.
    at.button(key="editor_next").click().run()
    assert at.session_state["editor_current_index"] == 1
    at.button(key=f"edit_task_{tasks1[1].id}").click().run()
    assert at.session_state["edit_task_open"] == tasks1[1].id

    # Switch the edited assignment mid-session.
    at.session_state["edit_assignment_id"] = a2.id
    at.run()

    assert not at.exception
    assert at.session_state["editor_assignment_id"] == a2.id
    assert at.session_state["editor_current_index"] == 0
    assert "edit_task_open" not in at.session_state


def test_editor_empty_assignment_still_offers_add_task(portal_env) -> None:
    """An assignment with no tasks shows the empty info + ➕ Add task (so the first task can be created)."""
    assignment, _ = _seed_assignment(n_tasks=0)

    at = _open_editor(assignment.id)

    assert not at.exception
    assert any("No tasks in this assignment yet" in info.value for info in at.info)
    assert any(b.key == "editor_add_task" for b in at.button)


def test_add_task_button_opens_create_dialog(portal_env) -> None:
    """➕ Add task opens the shared task-creation dialog for the assignment."""
    assignment, _ = _seed_assignment(n_tasks=1)

    at = _open_editor(assignment.id)
    at.button(key="editor_add_task").click().run()

    assert not at.exception
    assert at.session_state["create_task_open"] == assignment.id
    # The authoring form renders inside the dialog.
    assert _find_text_input(at, "topic") is not None
    assert _find_button(at, "🎯 Generate") is not None


def _generate_and_save_task(
    at: AppTest, assignment_id: int, topic: str, manual: str, task: GeneratedTask
) -> None:
    """Open the create dialog (via ➕ Add task), generate ``task``, and save it."""
    at.button(key="editor_add_task").click().run()
    _find_text_input(at, "topic").set_value(topic).run()
    _find_text_input(at, "subtopics").set_value(manual).run()
    with patch.object(
        MathProblemGenerator, "generate", new=AsyncMock(return_value=task)
    ):
        _find_button(at, "🎯 Generate").click().run()
    save = _find_button(at, "Save")
    assert save is not None, "expected the Save button in the dialog"
    save.click().run()


def test_add_task_generates_and_saves_then_jumps_to_it(portal_env) -> None:
    """Saving a generated task persists it and the carousel jumps to the new task."""
    assignment, _ = _seed_assignment(n_tasks=1)

    at = _open_editor(assignment.id, timeout=20)
    new_task = GeneratedTask(
        topic="Geometry",
        text="Triangle area?",
        complexity="easy",
        correct_answer="6",
        solution="Half base times height.",
    )
    _generate_and_save_task(at, assignment.id, "Geometry", "triangles", new_task)

    assert not at.exception
    tasks = get_storage().list_assignment_tasks(assignment.id)
    assert len(tasks) == 2
    assert tasks[-1].text == "Triangle area?"
    # The carousel jumped to the newly added (last) task.
    assert at.session_state["editor_current_index"] == 1
    assert "editor_jump_to_last" not in at.session_state
    marks = [m.value for m in at.markdown]
    assert any("Task 2 of 2" in m for m in marks)
    assert any("Triangle area?" in m for m in marks)


def test_remove_task_button_opens_confirm_dialog(portal_env) -> None:
    """The per-slide 🗑 Remove task button opens the confirmation dialog."""
    assignment, tasks = _seed_assignment(n_tasks=1)

    at = _open_editor(assignment.id)
    at.button(key=f"remove_task_{tasks[0].id}").click().run()

    assert not at.exception
    assert at.session_state["remove_task_open"] == tasks[0].id
    assert any(b.key == "remove_task_confirm" for b in at.button)
    assert any(b.key == "remove_task_cancel" for b in at.button)


def test_remove_task_confirm_removes_and_deletes_unshared(portal_env) -> None:
    """Confirming the dialog unlinks the task and deletes it (no other references)."""
    assignment, tasks = _seed_assignment(n_tasks=1)

    at = _open_editor(assignment.id)
    at.button(key=f"remove_task_{tasks[0].id}").click().run()
    at.button(key="remove_task_confirm").click().run()

    assert not at.exception
    assert get_storage().list_assignment_tasks(assignment.id) == []
    assert get_storage().get_task(tasks[0].id) is None
    # Removing the only task drops the page to the empty state.
    assert any("No tasks in this assignment yet" in info.value for info in at.info)


def test_remove_task_cancel_keeps_task(portal_env) -> None:
    """Cancel closes the dialog without unlinking the task."""
    assignment, tasks = _seed_assignment(n_tasks=1)

    at = _open_editor(assignment.id)
    at.button(key=f"remove_task_{tasks[0].id}").click().run()
    at.button(key="remove_task_cancel").click().run()

    assert not at.exception
    assert [t.id for t in get_storage().list_assignment_tasks(assignment.id)] == [tasks[0].id]
    assert get_storage().get_task(tasks[0].id) is not None
    assert "remove_task_open" not in at.session_state


def test_remove_shared_task_keeps_it_for_other_assignment(portal_env) -> None:
    """Removing a task shared with another assignment leaves it intact there."""
    a1, tasks = _seed_assignment(title="Alpha", n_tasks=1)
    storage = get_storage()
    a2 = storage.create_assignment(
        title="Beta", created_by=storage.get_user_by_username("boss").id
    )
    storage.add_task_to_assignment(a2.id, tasks[0].id)

    at = _open_editor(a1.id)
    at.button(key=f"remove_task_{tasks[0].id}").click().run()
    at.button(key="remove_task_confirm").click().run()

    assert not at.exception
    assert get_storage().get_task(tasks[0].id) is not None
    assert get_storage().list_assignment_tasks(a1.id) == []
    assert [t.id for t in get_storage().list_assignment_tasks(a2.id)] == [tasks[0].id]
