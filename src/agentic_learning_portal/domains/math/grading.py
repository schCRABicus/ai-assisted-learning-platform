"""Automatic grading of a student's attempt at a math assignment.

Grading is deliberately local and deterministic: the student's answer is compared
to the task's ``correct_answer`` with the same helper the verification judges use
(``domains/math/_comparison._answers_match``), so ``"4"``, ``"4.0"`` and ``" 4 "``
all match. No LLM call and no network — a submitted attempt is scored on the spot,
in the same run that submitted it.

A task the student left blank grades as incorrect; nothing is skipped, so every
task of the assignment ends up with exactly one scored result row.
"""

from __future__ import annotations

from agentic_learning_portal.domains.math._comparison import _answers_match
from agentic_learning_portal.storage import AttemptResult, Storage

CORRECT_SCORE = 1.0
INCORRECT_SCORE = 0.0


def grade_attempt(storage: Storage, attempt_id: int) -> list[AttemptResult]:
    """Grade every task of ``attempt_id`` and return the scored results.

    Reads the attempt's assignment tasks and whatever answers were saved, then
    upserts one scored result per task (``record_result`` replaces a task's
    previous row, so grading overwrites the ungraded answer saved earlier).
    Raises ``ValueError`` when no attempt with ``attempt_id`` exists.
    """
    attempt = storage.get_attempt(attempt_id)
    if attempt is None:
        raise ValueError(f"No attempt with id {attempt_id}")

    answered = {
        result.task_id: result.given_answer
        for result in storage.list_results(attempt_id=attempt_id)
    }

    graded: list[AttemptResult] = []
    for task in storage.list_assignment_tasks(attempt.assignment_id):
        given = answered.get(task.id)
        is_correct = _answers_match(task.correct_answer, given)
        graded.append(
            storage.record_result(
                attempt_id,
                task.id,
                given_answer=given,
                expected_answer=str(task.correct_answer),
                is_correct=is_correct,
                score=CORRECT_SCORE if is_correct else INCORRECT_SCORE,
                detail="Correct." if is_correct else "Incorrect.",
            )
        )
    return graded
