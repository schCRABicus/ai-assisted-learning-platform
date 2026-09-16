"""Attempt-flow rules shared by the student views.

Taking an assignment is a small state machine over a student's attempts, and both
the student list page and the take page need to ask the same questions about it:
is there an attempt in progress, how many attempts is this student entitled to,
and may they open a new one? Keeping the rule here means the two pages can't drift
apart on it — which matters, because the entitlement is the only thing standing
between a submitted assignment and a student re-taking it.

Related: ``domains/math/grading.grade_attempt`` scores a submitted attempt.
"""

from __future__ import annotations

from agentic_learning_portal.storage import Assignment, Attempt, Storage

# Statuses an ``Attempt`` can hold (see ``storage.models.AttemptStatus``).
IN_PROGRESS = "in_progress"
COMPLETED = "completed"


def student_attempts(
    storage: Storage, assignment_id: int, student_id: int
) -> list[Attempt]:
    """Return the student's attempts at an assignment, oldest first."""
    return storage.list_attempts(assignment_id=assignment_id, student_id=student_id)


def active_attempt(attempts: list[Attempt]) -> Attempt | None:
    """Return the attempt the student is still taking, or ``None``.

    Only one attempt is normally in progress; if several somehow are, the newest
    wins, matching ``attempts`` being ordered oldest first.
    """
    return next((a for a in reversed(attempts) if a.status == IN_PROGRESS), None)


def attempts_allowed(assignment: Assignment) -> int:
    """Return how many attempts the student may hold: the first plus any granted."""
    return 1 + assignment.extra_attempts


def can_start_attempt(assignment: Assignment, attempts: list[Attempt]) -> bool:
    """Return whether the student may open a fresh attempt.

    False while an attempt is still in progress (the student resumes that one)
    and once they have used every attempt they are entitled to — which is what
    locks a submitted assignment. Granting another attempt raises the
    entitlement, and since nothing is ever decremented the check can't drift.
    """
    return (
        active_attempt(attempts) is None
        and len(attempts) < attempts_allowed(assignment)
    )


def attempt_score(storage: Storage, attempt: Attempt) -> tuple[int, int, float | None]:
    """Summarize an attempt's results as ``(correct, graded, average_score)``.

    ``graded`` counts only results that carry a verdict, so an ungraded attempt
    reports ``(0, 0, None)``; the average is the mean of the per-task scores
    (1.0 per correct answer), i.e. the fraction of the assignment answered right.
    """
    results = storage.list_results(attempt_id=attempt.id)
    graded = [r for r in results if r.is_correct is not None]
    correct = sum(1 for r in graded if r.is_correct)
    scores = [r.score for r in results if r.score is not None]
    average = sum(scores) / len(scores) if scores else None
    return correct, len(graded), average
