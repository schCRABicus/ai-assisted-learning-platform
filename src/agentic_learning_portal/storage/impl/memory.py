from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Sequence

from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.storage.api import Storage
from agentic_learning_portal.storage.models import (
    Assignment,
    Attempt,
    AttemptResult,
    LazyTaskList,
    Role,
    RoleName,
    Task,
    User,
)
from agentic_learning_portal.storage.security import hash_password, verify_password

# The fixed set of roles every backend seeds (mirrors the 0001_initial
# migration's ``INSERT INTO roles ...``).
_BUILTIN_ROLES = [
    Role(id=1, name="admin"),
    Role(id=2, name="teacher"),
    Role(id=3, name="student"),
]


def _now() -> str:
    """Current UTC time as an ISO-8601 string (patchable in tests)."""
    return datetime.now(timezone.utc).isoformat()


def _parse_answer(raw: str) -> str | int | float:
    """Best-effort coercion of a stored answer back to its original type.

    ``correct_answer`` may be an int, float, or string; the storage layer keeps
    them as text, so on read we try to recover the numeric form. Mirrors
    ``storage/sqlite.py`` so both backends return identical values.
    """
    for cast in (int, float):
        try:
            return cast(raw)
        except (ValueError, TypeError):
            continue
    return raw


class InMemoryStorage(Storage):
    """Hash-table (dict-backed) :class:`Storage` for tests and throwaway use.

    Implements the same contract as :class:`SqliteStorage` — seeded roles,
    salted-scrypt password hashes, the ``.env`` admin seed, and the same
    ``ValueError`` / reference-check semantics — but keeps everything in plain
    dicts instead of a SQLite database. Construction is instant (no yoyo
    migrations, no reverse-DNS log lookups).

    Data lives on the instance only: unlike file-backed ``SqliteStorage``,
    instances never share state, and nothing survives a process exit. The portal
    accesses storage from a single script-run thread, so this is transparent in
    practice; it also gives tests clean isolation (fresh instance == empty
    storage). Access is guarded by a re-entrant lock for safety.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._roles: dict[int, Role] = {}
        self._users: dict[int, User] = {}
        self._tasks: dict[int, Task] = {}
        self._assignments: dict[int, Assignment] = {}
        # assignment_id -> list of (position, task_id); ordering mirrors the
        # sqlite ORDER BY position, task_id.
        self._assignment_tasks: dict[int, list[tuple[int, int]]] = {}
        self._attempts: dict[int, Attempt] = {}
        self._attempt_results: dict[int, AttemptResult] = {}

        self._next_user_id = 1
        self._next_task_id = 1
        self._next_assignment_id = 1
        self._next_attempt_id = 1
        self._next_result_id = 1

        with self._lock:
            for role in _BUILTIN_ROLES:
                self._roles[role.id] = role
            # Mirror SqliteStorage.__init__: provision the bootstrap admin
            # (admin + teacher) from .env when the vars are set.
            self.seed_admin_from_env()

    def close(self) -> None:
        """Drop all data (the dict equivalent of closing a memory DB)."""
        with self._lock:
            self._roles.clear()
            self._users.clear()
            self._tasks.clear()
            self._assignments.clear()
            self._assignment_tasks.clear()
            self._attempts.clear()
            self._attempt_results.clear()

    # --- internal helpers ------------------------------------------------------

    def _get_user(self, user_id: int) -> User:
        user = self._users.get(user_id)
        if user is None:
            raise ValueError(f"No user with id {user_id}")
        return user

    def _get_task(self, task_id: int) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"No task with id {task_id}")
        return task

    def _get_assignment(self, assignment_id: int) -> Assignment:
        assignment = self._assignments.get(assignment_id)
        if assignment is None:
            raise ValueError(f"No assignment with id {assignment_id}")
        return assignment

    def _get_attempt(self, attempt_id: int) -> Attempt:
        attempt = self._attempts.get(attempt_id)
        if attempt is None:
            raise ValueError(f"No attempt with id {attempt_id}")
        return attempt

    def _role_id(self, name: RoleName) -> int:
        for role in self._roles.values():
            if role.name == name:
                return role.id
        raise ValueError(f"Unknown role: {name}")

    # --- roles ---------------------------------------------------------------

    def list_roles(self) -> list[Role]:
        with self._lock:
            return sorted(self._roles.values(), key=lambda r: r.id)

    def get_role(self, name: RoleName) -> Role | None:
        with self._lock:
            return next((r for r in self._roles.values() if r.name == name), None)

    # --- users ---------------------------------------------------------------

    def create_user(
        self,
        username: str,
        roles: RoleName | Sequence[RoleName],
        *,
        email: str | None = None,
        password: str | None = None,
    ) -> User:
        role_names = [roles] if isinstance(roles, str) else list(roles)
        role_names = sorted(set(role_names))
        with self._lock:
            # Resolve every role before storing so an unknown role leaves no
            # partially-created user behind.
            [self._role_id(r) for r in role_names]
            if username in {u.username for u in self._users.values()}:
                raise ValueError(f"Username already exists: {username}")
            user_id = self._next_user_id
            self._next_user_id += 1
            user = User(
                id=user_id,
                username=username,
                roles=role_names,
                email=email,
                password_hash=hash_password(password) if password else None,
                created_at=_now(),
            )
            self._users[user_id] = user
            return user

    def get_user(self, user_id: int) -> User | None:
        with self._lock:
            return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> User | None:
        with self._lock:
            return next((u for u in self._users.values() if u.username == username), None)

    def list_users(self, *, role: RoleName | None = None) -> list[User]:
        with self._lock:
            users = sorted(self._users.values(), key=lambda u: u.id)
            if role is not None:
                users = [u for u in users if role in u.roles]
            return users

    def add_role(self, user_id: int, role: RoleName) -> User:
        with self._lock:
            user = self._get_user(user_id)
            role_id = self._role_id(role)
            if role not in user.roles:
                user = user.model_copy(update={"roles": sorted(user.roles + [role])})
                self._users[user_id] = user
            return user

    def remove_role(self, user_id: int, role: RoleName) -> User:
        with self._lock:
            user = self._get_user(user_id)
            self._role_id(role)
            user = user.model_copy(
                update={"roles": sorted(r for r in user.roles if r != role)}
            )
            self._users[user_id] = user
            return user

    # --- passwords & credentials ---------------------------------------------

    def set_password(self, user_id: int, password: str) -> User:
        """Set (or reset) the user's password hash and return the user."""
        with self._lock:
            user = self._get_user(user_id)
            user = user.model_copy(update={"password_hash": hash_password(password)})
            self._users[user_id] = user
            return user

    def verify_credentials(self, username: str, password: str) -> User | None:
        """Return the user if ``password`` matches their stored hash, else ``None``."""
        user = self.get_user_by_username(username)
        if user is None or not user.password_hash:
            return None
        if verify_password(password, user.password_hash):
            return user
        return None

    def seed_admin_from_env(self) -> User | None:
        """Create the initial admin (admin + teacher) from ``ADMIN_USERNAME`` /
        ``ADMIN_PASSWORD`` env vars.

        Same semantics as ``SqliteStorage.seed_admin_from_env``: idempotent,
        skipped when either var is unset, backfills a missing hash, never
        overwrites an existing one.
        """
        username = os.getenv("ADMIN_USERNAME")
        password = os.getenv("ADMIN_PASSWORD")
        if not username or not password:
            return None
        with self._lock:
            existing = self.get_user_by_username(username)
            if existing is not None:
                if existing.password_hash is None:
                    return self.set_password(existing.id, password)
                return None
            return self.create_user(username, ["admin", "teacher"], password=password)

    # --- tasks ---------------------------------------------------------------

    def create_task(self, task: GeneratedTask) -> Task:
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            stored = Task(
                id=task_id,
                topic=task.topic,
                text=task.text,
                complexity=task.complexity,
                correct_answer=str(task.correct_answer),
                solution=task.solution,
            )
            self._tasks[task_id] = stored
            return self._parse_stored_task(stored)

    def _parse_stored_task(self, stored: Task) -> Task:
        """Return ``stored`` with ``correct_answer`` coerced back to number/str."""
        return stored.model_copy(
            update={"correct_answer": _parse_answer(stored.correct_answer)}
        )

    def get_task(self, task_id: int) -> Task | None:
        with self._lock:
            stored = self._tasks.get(task_id)
            return self._parse_stored_task(stored) if stored is not None else None

    def list_tasks(self) -> list[Task]:
        with self._lock:
            return [
                self._parse_stored_task(t) for t in sorted(self._tasks.values(), key=lambda t: t.id)
            ]

    # --- assignments ---------------------------------------------------------

    def create_assignment(
        self,
        title: str,
        created_by: int,
        *,
        assigned_to: int | None = None,
    ) -> Assignment:
        with self._lock:
            self._get_user(created_by)
            if assigned_to is not None:
                self._get_user(assigned_to)
            assignment_id = self._next_assignment_id
            self._next_assignment_id += 1
            assignment = Assignment(
                id=assignment_id,
                title=title,
                created_by=created_by,
                assigned_to=assigned_to,
                created_at=_now(),
            )
            self._assignments[assignment_id] = assignment
        return self._with_tasks(assignment)

    def _with_tasks(self, assignment: Assignment) -> Assignment:
        """Return ``assignment`` with a fresh lazy ``tasks`` adapter."""
        with self._lock:
            count = len(self._assignment_tasks.get(assignment.id, []))
        return assignment.model_copy(
            update={
                "tasks": LazyTaskList(
                    count, lambda aid=assignment.id: self.list_assignment_tasks(aid)
                )
            }
        )

    def get_assignment(self, assignment_id: int) -> Assignment | None:
        with self._lock:
            assignment = self._assignments.get(assignment_id)
        return self._with_tasks(assignment) if assignment is not None else None

    def list_assignments(
        self,
        *,
        created_by: int | None = None,
        assigned_to: int | None = None,
    ) -> list[Assignment]:
        with self._lock:
            assignments = sorted(self._assignments.values(), key=lambda a: a.id)
            if created_by is not None:
                assignments = [a for a in assignments if a.created_by == created_by]
            if assigned_to is not None:
                assignments = [a for a in assignments if a.assigned_to == assigned_to]
        return [self._with_tasks(a) for a in assignments]


    def _assignments_with_task(self, task_id: int) -> list[int] | None:
        with self._lock:
            return [assignment_id for assignment_id, tasks in self._assignment_tasks.items() if task_id in map(lambda pair: pair[1], tasks)]

    def delete_assignment(self, assignment_id: int) -> None:
        with self._lock:
            if not assignment_id in self._assignments:
                return None

            del self._assignments[assignment_id]

            if assignment_id in self._assignment_tasks:
                assignment_task_ids = [task_id for _, task_id in self._assignment_tasks[assignment_id]]
                for task_id in assignment_task_ids:
                    if self._assignments_with_task(task_id) == [assignment_id]:
                        del self._tasks[task_id]
                del self._assignment_tasks[assignment_id]

            return assignment_id

    def add_task_to_assignment(
        self,
        assignment_id: int,
        task_id: int,
        *,
        position: int | None = None,
    ) -> None:
        with self._lock:
            self._get_assignment(assignment_id)
            self._get_task(task_id)
            links = self._assignment_tasks.setdefault(assignment_id, [])
            if any(t == task_id for _, t in links):
                raise ValueError(f"Task {task_id} already in assignment {assignment_id}")
            if position is None:
                position = max((p for p, _ in links), default=-1) + 1
            links.append((position, task_id))

    def list_assignment_tasks(self, assignment_id: int) -> list[Task]:
        with self._lock:
            links = sorted(
                self._assignment_tasks.get(assignment_id, []),
                key=lambda pair: (pair[0], pair[1]),
            )
            return [self._parse_stored_task(self._get_task(t)) for _, t in links]

    # --- attempts ------------------------------------------------------------

    def start_attempt(self, assignment_id: int, student_id: int) -> Attempt:
        with self._lock:
            self._get_assignment(assignment_id)
            self._get_user(student_id)
            attempt_id = self._next_attempt_id
            self._next_attempt_id += 1
            attempt = Attempt(
                id=attempt_id,
                assignment_id=assignment_id,
                student_id=student_id,
                status="in_progress",
                started_at=_now(),
            )
            self._attempts[attempt_id] = attempt
            return attempt

    def complete_attempt(self, attempt_id: int) -> Attempt:
        with self._lock:
            attempt = self._get_attempt(attempt_id)
            attempt = attempt.model_copy(
                update={"status": "completed", "completed_at": _now()}
            )
            self._attempts[attempt_id] = attempt
            return attempt

    def get_attempt(self, attempt_id: int) -> Attempt | None:
        with self._lock:
            return self._attempts.get(attempt_id)

    def list_attempts(
        self,
        *,
        student_id: int | None = None,
        assignment_id: int | None = None,
    ) -> list[Attempt]:
        with self._lock:
            attempts = sorted(self._attempts.values(), key=lambda a: a.id)
            if student_id is not None:
                attempts = [a for a in attempts if a.student_id == student_id]
            if assignment_id is not None:
                attempts = [a for a in attempts if a.assignment_id == assignment_id]
            return attempts

    # --- results -------------------------------------------------------------

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
        with self._lock:
            self._get_attempt(attempt_id)
            self._get_task(task_id)
            result_id = self._next_result_id
            self._next_result_id += 1
            result = AttemptResult(
                id=result_id,
                attempt_id=attempt_id,
                task_id=task_id,
                given_answer=given_answer,
                expected_answer=expected_answer,
                is_correct=is_correct,
                score=score,
                detail=detail,
            )
            self._attempt_results[result_id] = result
            return result

    def list_results(
        self,
        *,
        attempt_id: int | None = None,
        student_id: int | None = None,
    ) -> list[AttemptResult]:
        with self._lock:
            results = sorted(self._attempt_results.values(), key=lambda r: r.id)
            if attempt_id is not None:
                results = [r for r in results if r.attempt_id == attempt_id]
            if student_id is not None:
                results = [
                    r
                    for r in results
                    if self._attempts.get(r.attempt_id) is not None
                    and self._attempts[r.attempt_id].student_id == student_id
                ]
            return results