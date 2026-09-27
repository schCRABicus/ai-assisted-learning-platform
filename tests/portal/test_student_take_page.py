"""Tests for the assignment-taking page (views/student_take.py).

The page is reached from the student page with the assignment id in session state
(``take_assignment_id``), so each test seeds that and then signs in through the
login form, as a student would.

The journey the page exists for is covered end to end: start an attempt, save
progress, come back to the saved answers, submit, and find the assignment locked
with the graded results on screen — until an admin grants another attempt, which
hands out a fresh, blank one while the old results stay on record.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.auth import get_storage

TAKE_PAGE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "agentic_learning_portal"
    / "views"
    / "student_take.py"
)

STUDENT_PAGE = str(TAKE_PAGE.parent / "student.py")


def _student(username: str = "pupil"):
    return get_storage().create_user(username, "student", password="pw123")


def _assignment(
    *,
    assigned_to: int | None = None,
    tasks: int = 2,
    answers: list[str] | None = None,
):
    """Seed an assignment with ``tasks`` tasks, each with the given expected answer."""
    storage = get_storage()
    admin = storage.get_user_by_username("boss")
    assignment = storage.create_assignment(
        title="Algebra HW", created_by=admin.id, assigned_to=assigned_to
    )
    expected = answers or ["4"] * tasks
    task_ids = []
    for index in range(tasks):
        task = storage.create_task(
            GeneratedTask(
                topic=f"Topic {index + 1}",
                text=f"What is {index + 2} + 2?",
                complexity="easy",
                correct_answer=expected[index],
                solution="Add the numbers.",
            )
        )
        storage.add_task_to_assignment(assignment.id, task.id)
        task_ids.append(task.id)
    return assignment, task_ids


def _login_as(at: AppTest, username: str, password: str) -> None:
    at.text_input[0].set_value(username)
    at.text_input[1].set_value(password)
    at.button[0].click().run()


def _open_take_page(assignment_id: int | None, *, username: str = "pupil") -> AppTest:
    at = AppTest.from_file(str(TAKE_PAGE), default_timeout=10)
    if assignment_id is not None:
        at.session_state["take_assignment_id"] = assignment_id
    at.run()
    _login_as(at, username, "pw123")
    return at


def _answer_key(attempt_id: int, task_id: int) -> str:
    return f"take_answer_{attempt_id}_{task_id}"


def _attempts(assignment_id: int, student_id: int):
    return get_storage().list_attempts(
        assignment_id=assignment_id, student_id=student_id
    )


def _only_attempt(assignment_id: int, student_id: int):
    attempts = _attempts(assignment_id, student_id)
    assert len(attempts) == 1
    return attempts[0]


def _open_attempt(assignment_id: int, student_id: int):
    """Return the student's in-progress attempt (the one the form is bound to)."""
    open_attempts = [a for a in _attempts(assignment_id, student_id) if a.status == "in_progress"]
    assert len(open_attempts) == 1
    return open_attempts[0]


def _marks(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown]


def test_take_page_requires_authentication(portal_env) -> None:
    pupil = _student()
    assignment, _ = _assignment(assigned_to=pupil.id)

    at = AppTest.from_file(str(TAKE_PAGE), default_timeout=10)
    at.session_state["take_assignment_id"] = assignment.id
    at.run()

    assert not at.exception
    assert any(t.label == "Username" for t in at.text_input)
    assert not at.title


def test_take_page_without_a_selected_assignment(portal_env) -> None:
    _student()

    at = _open_take_page(None)

    assert not at.exception
    assert any("No assignment selected" in i.value for i in at.info)


def test_student_cannot_take_another_students_assignment(portal_env) -> None:
    """Session state is client-influenced, so the assignee is re-checked."""
    _student()
    classmate = _student("classmate")
    assignment, _ = _assignment(assigned_to=classmate.id)

    at = _open_take_page(assignment.id)

    assert not at.exception
    assert any("isn't available to you" in e.value for e in at.error)
    assert not at.text_input  # no answer form


def test_assignment_without_tasks_cannot_be_taken(portal_env) -> None:
    pupil = _student()
    assignment, _ = _assignment(assigned_to=pupil.id, tasks=0)

    at = _open_take_page(assignment.id)

    assert not at.exception
    assert any("has no tasks yet" in w.value for w in at.warning)
    assert get_storage().list_attempts(assignment_id=assignment.id) == []


def test_start_screen_offers_to_begin(portal_env) -> None:
    pupil = _student()
    assignment, _ = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)

    assert not at.exception
    assert at.title[0].value == "📝 Algebra HW"
    assert any("This assignment has **2 tasks**" in m for m in _marks(at))
    assert at.button(key="take_start").label == "▶️ Start assignment"


def test_starting_opens_the_answer_form(portal_env) -> None:
    pupil = _student()
    assignment, _ = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()

    assert not at.exception
    attempt = _only_attempt(assignment.id, pupil.id)
    assert attempt.status == "in_progress"
    assert len(at.text_input) == 2
    assert [t.label for t in at.text_input] == [
        "Your answer for task 1",
        "Your answer for task 2",
    ]
    assert at.button(key="take_save").label == "💾 Save progress"
    assert at.button(key="take_submit").label == "📤 Submit for grading"


def test_save_progress_persists_answers_and_keeps_attempt_open(portal_env) -> None:
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    attempt = _only_attempt(assignment.id, pupil.id)
    at.text_input(key=_answer_key(attempt.id, task_ids[0])).set_value("4")
    at.button(key="take_save").click().run()

    assert not at.exception
    assert any("Progress saved" in s.value for s in at.success)
    assert get_storage().get_attempt(attempt.id).status == "in_progress"  # type: ignore[union-attr]
    # Every task gets a row — the untouched one saved as blank, not skipped.
    saved = get_storage().list_results(attempt_id=attempt.id)
    assert [(r.task_id, r.given_answer) for r in saved] == [
        (task_ids[0], "4"),
        (task_ids[1], None),
    ]
    assert all(r.is_correct is None for r in saved)  # saved, not graded


def test_saved_answers_come_back_on_the_next_run(portal_env) -> None:
    """Leaving and reopening the page shows the answers saved earlier."""
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    attempt = _only_attempt(assignment.id, pupil.id)
    at.text_input(key=_answer_key(attempt.id, task_ids[1])).set_value("9")
    at.button(key="take_save").click().run()

    # A fresh page run, as if the student had closed and reopened the tab.
    reopened = _open_take_page(assignment.id)

    assert not reopened.exception
    assert reopened.text_input(key=_answer_key(attempt.id, task_ids[1])).value == "9"
    assert reopened.text_input(key=_answer_key(attempt.id, task_ids[0])).value == ""


def test_submitting_grades_locks_and_shows_results(portal_env) -> None:
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    attempt = _only_attempt(assignment.id, pupil.id)
    at.text_input(key=_answer_key(attempt.id, task_ids[0])).set_value("4")
    at.text_input(key=_answer_key(attempt.id, task_ids[1])).set_value("99")
    at.button(key="take_submit").click().run()

    assert not at.exception
    assert any("Submitted!" in s.value for s in at.success)
    attempt = _only_attempt(assignment.id, pupil.id)
    assert attempt.status == "completed"
    assert attempt.completed_at is not None

    results = {r.task_id: r for r in get_storage().list_results(attempt_id=attempt.id)}
    assert results[task_ids[0]].is_correct is True
    assert results[task_ids[0]].score == 1.0
    assert results[task_ids[1]].is_correct is False
    assert results[task_ids[1]].score == 0.0

    # The form is gone and the graded breakdown is on screen instead.
    assert not at.text_input
    marks = _marks(at)
    assert any("**📊 Score:** 1/2" in m for m in marks)
    assert any("✅ **Task 1**" in m for m in marks)
    assert any("❌ **Task 2**" in m for m in marks)
    assert any("**Your answer:** 4" in m for m in marks)
    assert any("**Correct answer:** `4`" in m for m in marks)


def test_submitted_assignment_is_locked_on_revisit(portal_env) -> None:
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    attempt = _only_attempt(assignment.id, pupil.id)
    at.text_input(key=_answer_key(attempt.id, task_ids[0])).set_value("4")
    at.button(key="take_submit").click().run()

    reopened = _open_take_page(assignment.id)

    assert not reopened.exception
    assert not reopened.text_input
    assert any("🔁 Start new attempt" not in b.label for b in reopened.button)
    assert any("used every attempt" in i.value for i in reopened.info)
    # Submitting again is not possible: the single attempt stays completed.
    assert _only_attempt(assignment.id, pupil.id).status == "completed"


def test_unanswered_task_grades_as_incorrect(portal_env) -> None:
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    attempt = _only_attempt(assignment.id, pupil.id)
    at.text_input(key=_answer_key(attempt.id, task_ids[0])).set_value("4")
    at.button(key="take_submit").click().run()

    rejected = [r for r in get_storage().list_results(attempt_id=attempt.id) if not r.is_correct]
    assert [r.task_id for r in rejected] == [task_ids[1]]
    assert rejected[0].given_answer is None


def test_granted_retake_offers_a_blank_attempt_and_keeps_the_results(portal_env) -> None:
    """A granted retake opens a fresh attempt, with the graded one still shown."""
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    first = _only_attempt(assignment.id, pupil.id)
    at.text_input(key=_answer_key(first.id, task_ids[0])).set_value("4")
    at.button(key="take_submit").click().run()

    get_storage().grant_extra_attempt(assignment.id)
    reopened = _open_take_page(assignment.id)

    assert not reopened.exception
    # The graded attempt is still on record and rendered.
    assert any("**📊 Score:** 1/2" in m for m in _marks(reopened))
    assert any("granted you another attempt" in w.value for w in reopened.warning)

    reopened.button(key="take_start_new").click().run()

    assert not reopened.exception
    attempts = get_storage().list_attempts(
        assignment_id=assignment.id, student_id=pupil.id
    )
    assert [a.status for a in attempts] == ["completed", "in_progress"]
    retake = attempts[-1]
    assert retake.id != first.id
    # Fresh and blank: the previous attempt's answers are not carried over.
    assert len(reopened.text_input) == 2
    assert [t.value for t in reopened.text_input] == ["", ""]
    assert get_storage().list_results(attempt_id=retake.id) == []


def test_retake_does_not_touch_the_first_attempts_results(portal_env) -> None:
    """Grading the retake leaves the earlier attempt's grades alone."""
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    first = _only_attempt(assignment.id, pupil.id)
    at.text_input(key=_answer_key(first.id, task_ids[0])).set_value("4")
    at.button(key="take_submit").click().run()

    get_storage().grant_extra_attempt(assignment.id)
    second = _open_take_page(assignment.id)
    second.button(key="take_start_new").click().run()
    retake = _open_attempt(assignment.id, pupil.id).id
    second.text_input(key=_answer_key(retake, task_ids[0])).set_value("nonsense")
    second.button(key="take_submit").click().run()

    assert not second.exception
    storage = get_storage()
    first_results = {r.task_id: r.is_correct for r in storage.list_results(attempt_id=first.id)}
    retake_results = {r.task_id: r.is_correct for r in storage.list_results(attempt_id=retake)}
    assert first_results[task_ids[0]] is True
    assert retake_results[task_ids[0]] is False
    # Both attempts are listed, newest first.
    assert [a.status for a in storage.list_attempts(assignment_id=assignment.id)] == [
        "completed",
        "completed",
    ]


def test_results_are_upserted_not_duplicated_across_saves(portal_env) -> None:
    """Saving twice then submitting leaves one result row per task."""
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    attempt = _only_attempt(assignment.id, pupil.id)
    at.text_input(key=_answer_key(attempt.id, task_ids[0])).set_value("1")
    at.button(key="take_save").click().run()
    at.text_input(key=_answer_key(attempt.id, task_ids[0])).set_value("4")
    at.button(key="take_save").click().run()
    at.button(key="take_submit").click().run()

    assert not at.exception
    results = get_storage().list_results(attempt_id=attempt.id)
    assert len(results) == 2
    assert len({r.task_id for r in results}) == 2


def test_back_button_returns_to_the_student_page(portal_env) -> None:
    pupil = _student()
    assignment, _ = _assignment(assigned_to=pupil.id)

    with patch("streamlit.switch_page") as switch:
        at = _open_take_page(assignment.id)
        at.button(key="take_back").click().run()

    assert not at.exception
    switch.assert_called_once_with(STUDENT_PAGE)


# --- the solution box ----------------------------------------------------------


def _solution_key(attempt_id: int, task_id: int) -> str:
    return f"take_solution_{attempt_id}_{task_id}"


def test_solution_box_renders_beside_each_answer(portal_env) -> None:
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    attempt = _only_attempt(assignment.id, pupil.id)

    assert not at.exception
    # The working is multi-line, so it gets a text area of its own per task —
    # the answers stay text inputs.
    assert len(at.text_input) == 2
    assert [t.key for t in at.text_area] == [
        _solution_key(attempt.id, task_ids[0]),
        _solution_key(attempt.id, task_ids[1]),
    ]
    assert [t.label for t in at.text_area] == [
        "✍️ Your solution (optional) for task 1",
        "✍️ Your solution (optional) for task 2",
    ]


def test_save_progress_persists_the_solution(portal_env) -> None:
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    attempt = _only_attempt(assignment.id, pupil.id)
    at.text_input(key=_answer_key(attempt.id, task_ids[0])).set_value("4")
    at.text_area(key=_solution_key(attempt.id, task_ids[0])).set_value("2 + 2\n= 4")
    at.button(key="take_save").click().run()

    assert not at.exception
    assert any("Progress saved" in s.value for s in at.success)
    saved = get_storage().list_results(attempt_id=attempt.id)
    assert saved[0].given_solution == "2 + 2\n= 4"
    # Still ungraded — saving is not submitting.
    assert saved[0].is_correct is None


def test_a_blank_solution_is_allowed(portal_env) -> None:
    """The solution is optional: an answer alone saves and grades fine."""
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    attempt = _only_attempt(assignment.id, pupil.id)
    at.text_input(key=_answer_key(attempt.id, task_ids[0])).set_value("4")
    at.button(key="take_save").click().run()

    assert not at.exception
    saved = get_storage().list_results(attempt_id=attempt.id)
    assert saved[0].given_answer == "4"
    assert saved[0].given_solution is None


def test_a_cleared_solution_really_clears(portal_env) -> None:
    """Emptying the box and saving clears the stored working — it isn't sticky."""
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    attempt = _only_attempt(assignment.id, pupil.id)
    at.text_area(key=_solution_key(attempt.id, task_ids[0])).set_value("wrong idea")
    at.button(key="take_save").click().run()
    at.text_area(key=_solution_key(attempt.id, task_ids[0])).set_value("")
    at.button(key="take_save").click().run()

    assert not at.exception
    assert get_storage().list_results(attempt_id=attempt.id)[0].given_solution is None


def test_saved_solutions_come_back_on_the_next_run(portal_env) -> None:
    """Leaving and reopening the page shows the working saved earlier."""
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    attempt = _only_attempt(assignment.id, pupil.id)
    at.text_area(key=_solution_key(attempt.id, task_ids[1])).set_value("9 - 7 = 2")
    at.button(key="take_save").click().run()

    # A fresh page run, as if the student had closed and reopened the tab.
    reopened = _open_take_page(assignment.id)

    assert not reopened.exception
    assert reopened.text_area(key=_solution_key(attempt.id, task_ids[1])).value == "9 - 7 = 2"
    assert reopened.text_area(key=_solution_key(attempt.id, task_ids[0])).value == ""


def test_submitting_preserves_the_solution(portal_env) -> None:
    """Grading rewrites each result row — the working has to survive it."""
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    attempt = _only_attempt(assignment.id, pupil.id)
    at.text_input(key=_answer_key(attempt.id, task_ids[0])).set_value("4")
    at.text_area(key=_solution_key(attempt.id, task_ids[0])).set_value("2 + 2 = 4")
    at.button(key="take_submit").click().run()

    assert not at.exception
    graded = get_storage().list_results(attempt_id=attempt.id)
    assert graded[0].is_correct is True
    assert graded[0].given_solution == "2 + 2 = 4"


def test_graded_breakdown_shows_the_students_solution(portal_env) -> None:
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    attempt = _only_attempt(assignment.id, pupil.id)
    at.text_input(key=_answer_key(attempt.id, task_ids[0])).set_value("4")
    at.text_area(key=_solution_key(attempt.id, task_ids[0])).set_value("2 + 2 = 4")
    at.button(key="take_submit").click().run()

    assert not at.exception
    assert any("✍️ Your solution:" in m and "2 + 2 = 4" in m for m in _marks(at))
    # The task that was left blank says so rather than showing an empty line.
    assert any("_(not given)_" in m for m in _marks(at))


def test_grading_does_not_read_the_solution_as_an_answer(portal_env) -> None:
    """The working is never graded: a right answer with nonsense working passes."""
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    attempt = _only_attempt(assignment.id, pupil.id)
    at.text_input(key=_answer_key(attempt.id, task_ids[0])).set_value("4")
    at.text_area(key=_solution_key(attempt.id, task_ids[0])).set_value("no idea, guessed")
    at.button(key="take_submit").click().run()

    assert not at.exception
    graded = get_storage().list_results(attempt_id=attempt.id)
    assert graded[0].is_correct is True
    assert graded[0].given_answer == "4"


def test_a_new_attempt_starts_with_a_blank_solution(portal_env) -> None:
    """A granted retake shows blank working — the first attempt's text must not leak."""
    pupil = _student()
    assignment, task_ids = _assignment(assigned_to=pupil.id)

    at = _open_take_page(assignment.id)
    at.button(key="take_start").click().run()
    first = _only_attempt(assignment.id, pupil.id)
    at.text_input(key=_answer_key(first.id, task_ids[0])).set_value("4")
    at.text_area(key=_solution_key(first.id, task_ids[0])).set_value("first go")
    at.button(key="take_submit").click().run()
    get_storage().grant_extra_attempt(assignment.id)

    retake_at = _open_take_page(assignment.id)
    retake_at.button(key="take_start_new").click().run()
    retake = _open_attempt(assignment.id, pupil.id)

    assert not retake_at.exception
    assert retake.id != first.id
    assert [t.value for t in retake_at.text_area] == ["", ""]
