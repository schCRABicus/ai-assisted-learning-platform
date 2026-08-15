from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

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
    ) -> User:
        """Create and return a new user holding one or more roles.

        When ``password`` is given it is stored as a salted hash (never as
        plaintext); leave it ``None`` for users who don't log in yet.
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
    def set_password(self, user_id: int, password: str) -> User:
        """Set (or reset) the user's password hash and return the user."""

    @abstractmethod
    def verify_credentials(self, username: str, password: str) -> User | None:
        """Return the user if ``password`` matches their stored hash, else ``None``."""

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

    # --- assignments ---------------------------------------------------------

    @abstractmethod
    def create_assignment(
        self,
        title: str,
        created_by: int,
        *,
        assigned_to: int | None = None,
    ) -> Assignment:
        """Create and return a new assignment."""

    @abstractmethod
    def get_assignment(self, assignment_id: int) -> Assignment | None:
        """Return the assignment with ``assignment_id``, or ``None`` if missing."""

    @abstractmethod
    def list_assignments(
        self,
        *,
        created_by: int | None = None,
        assigned_to: int | None = None,
    ) -> list[Assignment]:
        """Return assignments, optionally filtered by creator or assignee."""

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
    def list_assignment_tasks(self, assignment_id: int) -> list[Task]:
        """Return the tasks of an assignment in order."""

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
    ) -> list[Attempt]:
        """Return attempts, optionally filtered by student or assignment."""

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
        """Record a scored answer for one task of an attempt."""

    @abstractmethod
    def list_results(
        self,
        *,
        attempt_id: int | None = None,
        student_id: int | None = None,
    ) -> list[AttemptResult]:
        """Return results, optionally filtered by attempt or student."""