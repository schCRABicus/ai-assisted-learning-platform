from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Sequence

from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.storage.models import (
    Assignment,
    Attempt,
    AttemptResult,
    Role,
    RoleName,
    Task,
    User,
)


class Storage(ABC):
    """Persistence contract for the portal.

    A concrete backend (e.g. ``SqliteStorage``) implements these methods. All
    methods are synchronous: the current backends are local and fast, and
    callers on the async generation pipeline can wrap them in ``asyncio.to_thread``
    if ever needed.
    """

    # --- roles ---------------------------------------------------------------

    @abstractmethod
    def list_roles(self) -> list[Role]:
        """Return all roles in the system."""

    @abstractmethod
    def get_role(self, name: RoleName) -> Role | None:
        """Return the role with ``name``, or ``None`` if unknown."""

    # --- users ---------------------------------------------------------------

    @abstractmethod
    def create_user(
        self,
        username: str,
        roles: RoleName | Sequence[RoleName],
        *,
        email: str | None = None,
        password: str | None = None,
        email_verified: bool = True,
    ) -> User:
        """Create and return a new user holding one or more roles.

        When ``password`` is given it is stored as a salted hash (never as
        plaintext); leave it ``None`` for users who don't log in yet. An
        invited user is created with ``email_verified=False`` and no password,
        so it cannot sign in until the verification link is opened and a
        password set.
        """

    @abstractmethod
    def get_user(self, user_id: int) -> User | None:
        """Return the user with ``user_id``, or ``None`` if missing."""

    @abstractmethod
    def get_user_by_username(self, username: str) -> User | None:
        """Return the user with ``username``, or ``None`` if missing."""

    @abstractmethod
    def list_users(self, *, role: RoleName | None = None) -> list[User]:
        """Return users, optionally filtered by membership in ``role``."""

    @abstractmethod
    def add_role(self, user_id: int, role: RoleName) -> User:
        """Grant ``role`` to the user and return the updated user."""

    @abstractmethod
    def remove_role(self, user_id: int, role: RoleName) -> User:
        """Revoke ``role`` from the user and return the updated user."""

    @abstractmethod
    def update_user(
        self,
        user_id: int,
        *,
        username: str | None = None,
        email: str | None = None,
        roles: Sequence[RoleName] | None = None,
    ) -> User:
        """Update the editable fields of an existing user and return it.

        Only the fields given are changed; ``None`` leaves a field untouched.
        ``roles`` replaces the user's full role set (not a delta). Changing
        ``email`` clears ``email_verified`` and any pending verification token,
        since the address is the identity the invite link confirms. Raises
        ``ValueError`` when no user with ``user_id`` exists.
        """

    @abstractmethod
    def issue_verification_token(
        self,
        user_id: int,
        *,
        ttl_days: int = 7,
    ) -> str:
        """Generate and store a verification token for ``user_id``.

        Returns the raw token (the only place it ever exists) and records its
        hash plus an expiry ``ttl_days`` from now on the user. Re-issuing
        overwrites any previous token. Raises ``ValueError`` when no user with
        ``user_id`` exists.
        """

    @abstractmethod
    def get_user_by_verification_token(self, token: str) -> User | None:
        """Return the user whose pending token hashes to ``token``.

        Returns ``None`` when the token is unknown, the user is already
        verified, or the token has expired.
        """

    @abstractmethod
    def complete_email_verification(self, user_id: int) -> User:
        """Mark the user's email verified and clear its pending token.

        Raises ``ValueError`` when no user with ``user_id`` exists.
        """

    @abstractmethod
    def set_password(self, user_id: int, password: str) -> User:
        """Set (or reset) the user's password hash and return the user."""

    @abstractmethod
    def verify_credentials(self, username: str, password: str) -> User | None:
        """Return the user if ``password`` matches their stored hash, else ``None``.

        An unverified user (``email_verified=False``) never matches, so an
        invited account can't sign in before opening the verification link.
        """

    # --- tasks ---------------------------------------------------------------

    @abstractmethod
    def create_task(self, task: GeneratedTask) -> Task:
        """Persist a generated task and return it with its storage ``id``."""

    @abstractmethod
    def get_task(self, task_id: int) -> Task | None:
        """Return the task with ``task_id``, or ``None`` if missing."""

    @abstractmethod
    def list_tasks(self) -> list[Task]:
        """Return all persisted tasks."""

    @abstractmethod
    def update_task(
        self,
        task_id: int,
        *,
        topic: str | None = None,
        text: str | None = None,
        complexity: Literal["easy", "medium", "hard"] | None = None,
        correct_answer: str | int | float | None = None,
        solution: str | None = None,
    ) -> Task:
        """Update the editable fields of an existing task and return the updated task.

        Only the fields given are changed; ``None`` leaves a field untouched.
        Raises ``ValueError`` when no task with ``task_id`` exists.
        """

    # --- assignments ---------------------------------------------------------

    @abstractmethod
    def create_assignment(
        self,
        title: str,
        created_by: int,
        *,
        assigned_to: int | None = None,
    ) -> Assignment:
        """Create and return a new assignment (with an empty lazy ``tasks`` list)."""

    @abstractmethod
    def get_assignment(self, assignment_id: int) -> Assignment | None:
        """Return the assignment with ``assignment_id``, or ``None`` if missing.

        The returned ``Assignment.tasks`` is a lazy ``LazyTaskList``: its
        ``size`` (task count) is joined in when loading, while the task
        contents are only fetched on first access.
        """

    @abstractmethod
    def list_assignments(
        self,
        *,
        created_by: int | None = None,
        assigned_to: int | None = None,
    ) -> list[Assignment]:
        """Return assignments, optionally filtered by creator or assignee.

        Each assignment's ``tasks`` is a lazy ``LazyTaskList``: its ``size``
        (task count) is joined in without loading the tasks themselves; the
        task contents are only fetched on first access.
        """

    @abstractmethod
    def delete_assignment(self, assignment_id: int) -> int | None:
        """Delete an assignment and related tasks and return its ``id``, if exists.
        Otherwise, return ``None``.
        """

    # --- assignment management ----------------------------------------------------

    @abstractmethod
    def add_task_to_assignment(
        self,
        assignment_id: int,
        task_id: int,
        *,
        position: int | None = None,
    ) -> None:
        """Attach a task to an assignment at ``position`` (appends when omitted)."""

    @abstractmethod
    def assign_assignment(
        self,
        assignment_id: int,
        assign_to: int | None = None,
    ) -> None:
        """Assign the specified assignment to ``assign_to``, or ``None`` if missing."""

    @abstractmethod
    def grant_extra_attempt(self, assignment_id: int) -> Assignment:
        """Grant one more attempt on the assignment and return the updated assignment.

        Increments ``extra_attempts``, raising the student's entitlement to
        ``1 + extra_attempts`` attempts. Nothing is decremented when the extra
        attempt is used, so the entitlement can't drift. Raises ``ValueError``
        when no assignment with ``assignment_id`` exists.
        """

    @abstractmethod
    def list_assignment_tasks(self, assignment_id: int) -> list[Task]:
        """Return the tasks of an assignment in order."""

    @abstractmethod
    def remove_task_from_assignment(self, assignment_id: int, task_id: int) -> None:
        """Unlink ``task_id`` from ``assignment_id``'s task list.

        The task itself is deleted when no other assignment references it; a task
        still shared by another assignment survives. Raises ``ValueError`` when
        the assignment is missing or the task is not part of the assignment.
        """

    # --- attempts ------------------------------------------------------------

    @abstractmethod
    def start_attempt(self, assignment_id: int, student_id: int) -> Attempt:
        """Start (and return) a new in-progress attempt."""

    @abstractmethod
    def complete_attempt(self, attempt_id: int) -> Attempt:
        """Mark an attempt completed and return it."""

    @abstractmethod
    def get_attempt(self, attempt_id: int) -> Attempt | None:
        """Return the attempt with ``attempt_id``, or ``None`` if missing."""

    @abstractmethod
    def list_attempts(
        self,
        *,
        student_id: int | None = None,
        assignment_id: int | None = None,
        results_seen: bool | None = None,
    ) -> list[Attempt]:
        """Return attempts, optionally filtered by student, assignment, or seen state.

        ``results_seen`` narrows to attempts whose graded results the admin has
        (``True``) or hasn't (``False``) looked at yet — the "new results" panel
        asks for ``False``.
        """

    @abstractmethod
    def mark_results_seen(self, attempt_id: int) -> Attempt:
        """Mark the attempt's graded results as seen by the admin, and return it.

        Raises ``ValueError`` when no attempt with ``attempt_id`` exists.
        """

    # --- results -------------------------------------------------------------

    @abstractmethod
    def record_result(
        self,
        attempt_id: int,
        task_id: int,
        *,
        given_answer: str | None = None,
        expected_answer: str | None = None,
        is_correct: bool | None = None,
        score: float | None = None,
        detail: str = "",
    ) -> AttemptResult:
        """Record an answer for one task of an attempt, replacing any previous one.

        A student saves progress and is later graded through the same call, so
        the write is an upsert on ``(attempt_id, task_id)``: at most one row
        exists per task per attempt, and a second call overwrites it (grading
        fields included) rather than appending a duplicate.
        """

    @abstractmethod
    def list_results(
        self,
        *,
        attempt_id: int | None = None,
        student_id: int | None = None,
    ) -> list[AttemptResult]:
        """Return results, optionally filtered by attempt or student."""
