"""Tests for automatic grading of a submitted attempt.

Grading is pure local computation — the answer comparison is the same helper the
judges use — so these tests run against a real :class:`InMemoryStorage` rather
than a fake: the point is as much that the right rows land in storage (one scored
row per task, upserted over the saved answer) as that the verdicts are right.
"""

from __future__ import annotations

import pytest

from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.domains.math.grading import (
    CORRECT_SCORE,
    INCORRECT_SCORE,
    grade_attempt,
)
from agentic_learning_portal.storage import AttemptResult, InMemoryStorage


def _task(correct_answer: str | int | float = "12", topic: str = "Arithmetic") -> GeneratedTask:
    return GeneratedTask.model_validate(
        {
            "topic": topic,
            "text": "What is 3 times 4?",
            "complexity": "easy",
            "correct_answer": correct_answer,
            "solution": "Multiply the two numbers: 3 × 4 = 12.",
        }
    )


def _attempt_with_tasks(
    answers: dict[int, str | None],
    *,
    correct_answers: list[str | int | float] | None = None,
) -> tuple[InMemoryStorage, int, list[int]]:
    """Build an assignment of one task per entry in ``answers``, with a student attempt.

    ``answers`` maps a task index to the answer to save (``None`` = left blank);
    ``correct_answers`` overrides each task's expected answer by index. Returns the
    storage, the attempt id, and the task ids in assignment order.
    """
    storage = InMemoryStorage()
    teacher = storage.create_user("teacher1", "teacher")
    student = storage.create_user("student1", "student")
    assignment = storage.create_assignment("Quiz", teacher.id, assigned_to=student.id)

    expected = correct_answers or ["12"] * len(answers)
    task_ids = []
    for index in range(len(answers)):
        task = storage.create_task(_task(expected[index], topic=f"Topic {index}"))
        storage.add_task_to_assignment(assignment.id, task.id)
        task_ids.append(task.id)

    attempt = storage.start_attempt(assignment.id, student.id)
    for index, given in answers.items():
        if given is not None:
            storage.record_result(attempt.id, task_ids[index], given_answer=given)
    return storage, attempt.id, task_ids


def _result_for(storage: InMemoryStorage, attempt_id: int, task_id: int) -> AttemptResult:
    matches = [r for r in storage.list_results(attempt_id=attempt_id) if r.task_id == task_id]
    assert len(matches) == 1  # one row per task, no duplicates
    return matches[0]


def test_grade_attempt_scores_correct_and_incorrect() -> None:
    storage, attempt_id, task_ids = _attempt_with_tasks(
        {0: "12", 1: "7"}, correct_answers=["12", "12"]
    )

    graded = grade_attempt(storage, attempt_id)

    assert len(graded) == 2
    assert _result_for(storage, attempt_id, task_ids[0]).is_correct is True
    assert _result_for(storage, attempt_id, task_ids[1]).is_correct is False


def test_grade_attempt_sets_score_and_detail() -> None:
    storage, attempt_id, task_ids = _attempt_with_tasks({0: "12"})

    (graded,) = grade_attempt(storage, attempt_id)

    assert graded.score == CORRECT_SCORE == 1.0
    assert graded.detail == "Correct."
    assert graded.expected_answer == "12"
    assert graded.given_answer == "12"


def test_incorrect_answer_scores_zero() -> None:
    storage, attempt_id, _ = _attempt_with_tasks({0: "13"})

    (graded,) = grade_attempt(storage, attempt_id)

    assert graded.is_correct is False
    assert graded.score == INCORRECT_SCORE == 0.0
    assert graded.detail == "Incorrect."


def test_unanswered_task_grades_as_incorrect() -> None:
    storage, attempt_id, task_ids = _attempt_with_tasks({0: None})

    (graded,) = grade_attempt(storage, attempt_id)

    assert graded.is_correct is False
    assert graded.score == INCORRECT_SCORE
    assert graded.given_answer is None
    assert graded.expected_answer == "12"


def test_numeric_answers_match_across_notations() -> None:
    storage, attempt_id, _ = _attempt_with_tasks({0: "4.0"}, correct_answers=[4])

    (graded,) = grade_attempt(storage, attempt_id)

    assert graded.is_correct is True


def test_answers_match_despite_surrounding_whitespace() -> None:
    storage, attempt_id, _ = _attempt_with_tasks({0: "  12  "})

    (graded,) = grade_attempt(storage, attempt_id)

    assert graded.is_correct is True


def test_text_answers_match_case_insensitively() -> None:
    storage, attempt_id, _ = _attempt_with_tasks(
        {0: "third"}, correct_answers=["Third"]
    )

    (graded,) = grade_attempt(storage, attempt_id)

    assert graded.is_correct is True


def test_grading_replaces_the_saved_answer_row() -> None:
    """A task saved earlier keeps its one row, now carrying the grade."""
    storage, attempt_id, task_ids = _attempt_with_tasks({0: "12"})

    saved = storage.list_results(attempt_id=attempt_id)
    assert saved[0].is_correct is None  # saved, not graded yet

    graded = grade_attempt(storage, attempt_id)

    assert [r.id for r in graded] == [saved[0].id]
    assert len(storage.list_results(attempt_id=attempt_id)) == 1


def test_grading_twice_is_stable() -> None:
    storage, attempt_id, _ = _attempt_with_tasks({0: "12"})

    first = grade_attempt(storage, attempt_id)
    second = grade_attempt(storage, attempt_id)

    assert [r.id for r in second] == [r.id for r in first]
    assert len(storage.list_results(attempt_id=attempt_id)) == 1


def test_grading_a_retake_does_not_touch_the_earlier_attempt() -> None:
    storage, attempt_id, task_ids = _attempt_with_tasks({0: "12"})
    grade_attempt(storage, attempt_id)
    assignment_id = storage.get_attempt(attempt_id).assignment_id  # type: ignore[union-attr]
    student_id = storage.get_attempt(attempt_id).student_id  # type: ignore[union-attr]

    retake = storage.start_attempt(assignment_id, student_id)
    storage.record_result(retake.id, task_ids[0], given_answer="13")
    grade_attempt(storage, retake.id)

    assert _result_for(storage, attempt_id, task_ids[0]).is_correct is True
    assert _result_for(storage, retake.id, task_ids[0]).is_correct is False


def test_grade_attempt_missing_raises() -> None:
    storage = InMemoryStorage()

    with pytest.raises(ValueError, match="No attempt with id 999"):
        grade_attempt(storage, 999)


def test_grading_preserves_the_students_solution() -> None:
    """Grading rewrites the whole row, so it has to hand the working back in.

    ``record_result`` overwrites every column; if grading didn't pass the saved
    solution through, submitting would silently erase what the student wrote.
    """
    storage, attempt_id, task_ids = _attempt_with_tasks({0: "12"})
    storage.record_result(
        attempt_id,
        task_ids[0],
        given_answer="12",
        given_solution="3 × 4\n= 12",
    )

    grade_attempt(storage, attempt_id)

    graded = _result_for(storage, attempt_id, task_ids[0])
    assert graded.given_solution == "3 × 4\n= 12"
    assert graded.is_correct is True


def test_grading_clears_a_manual_adjustment() -> None:
    """A fresh auto-grade supersedes an override, so the marker goes back to false."""
    storage, attempt_id, task_ids = _attempt_with_tasks({0: "12"})
    storage.record_result(attempt_id, task_ids[0], given_answer="12")
    storage.adjust_result(attempt_id, task_ids[0], is_correct=False, score=0.25)

    grade_attempt(storage, attempt_id)

    graded = _result_for(storage, attempt_id, task_ids[0])
    assert graded.score_adjusted is False
    assert graded.is_correct is True
    assert graded.score == 1.0
