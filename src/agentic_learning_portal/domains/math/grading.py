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

    Reads the attempt's assignment tasks and whatever was saved, then upserts one
    scored result per task (``record_result`` replaces a task's previous row, so
    grading overwrites the ungraded answer saved earlier). The student's working
    is passed back through unchanged — it is never graded, and dropping it here
    would erase it. Because grading is authoritative, an existing manual
    override is superseded and ``score_adjusted`` goes back to false.

    Raises ``ValueError`` when no attempt with ``attempt_id`` exists.
    """
    attempt = storage.get_attempt(attempt_id)
    if attempt is None:
        raise ValueError(f"No attempt with id {attempt_id}")

    saved = {
        result.task_id: result
        for result in storage.list_results(attempt_id=attempt_id)
    }

    graded: list[AttemptResult] = []
    for task in storage.list_assignment_tasks(attempt.assignment_id):
        previous = saved.get(task.id)
        given = previous.given_answer if previous is not None else None
        # ``record_result`` overwrites every column, so the saved solution has to
        # be handed back to it or grading would silently clear the student's work.
        solution = previous.given_solution if previous is not None else None
        is_correct = _answers_match(task.correct_answer, given)
        graded.append(
            storage.record_result(
                attempt_id,
                task.id,
                given_answer=given,
                given_solution=solution,
                expected_answer=str(task.correct_answer),
                is_correct=is_correct,
                score=CORRECT_SCORE if is_correct else INCORRECT_SCORE,
                detail="Correct." if is_correct else "Incorrect.",
            )
        )
    return graded
