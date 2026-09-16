from __future__ import annotations

import getpass
import os
import secrets
import socket
import sqlite3
import threading
import uuid
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from yoyo import read_migrations
from yoyo.backends import SQLiteBackend
from yoyo.connections import parse_uri

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
from agentic_learning_portal.storage.security import (
    hash_password,
    hash_token,
    verify_password,
)

# Where the versioned SQL migrations live (applied by yoyo on construction).
MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


class _FastLogMixin:
    """Avoid yoyo's reverse-DNS ``socket.getfqdn()`` in migration logs.

    yoyo's ``DatabaseBackend.get_log_data`` records the fully-qualified
    hostname via ``socket.getfqdn()``, which performs a reverse-DNS lookup that
    can stall for seconds when the local hostname isn't resolvable (common on
    dev laptops). The hostname is only migration-log metadata, so the instant
    ``socket.gethostname()`` is enough. Mirrors yoyo's dict format from
    ``yoyo/backends/api.py``, minus the lookup.
    """

    def get_log_data(self, migration=None, operation="apply"):
        assert operation in {"apply", "rollback", "mark", "unmark"}
        return {
            "id": str(uuid.uuid1()),
            "migration_id": migration.id if migration else None,
            "migration_hash": migration.hash if migration else None,
            "username": getpass.getuser(),
            "hostname": socket.gethostname(),
            "created_at_utc": datetime.now(timezone.utc).replace(tzinfo=None),
            "operation": operation,
        }


class _NamedMemoryBackend(_FastLogMixin, SQLiteBackend):
    """yoyo SQLite backend bound to a private named shared-cache memory DB.

    yoyo's own ``connect()`` opens ``file::memory:?cache=shared``, which is a
    single process-global in-memory database shared by every connection — fine
    for one storage, but it would make two ``SqliteStorage`` instances share
    data. This backend instead connects every instance (and every migration
    copy) to ``file:<name>?mode=memory&cache=shared``, so each storage gets an
    isolated, truly in-memory database that lives only while its connections
    are open.
    """

    def __init__(
        self,
        uri,
        dbname: str,
        migration_table: str = "_yoyo_migration",
    ) -> None:
        self._dbname = dbname
        super().__init__(uri, migration_table)

    def connect(self, dburi) -> sqlite3.Connection:
        conn = sqlite3.connect(
            f"file:{self._dbname}?mode=memory&cache=shared",
            uri=True,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        conn.isolation_level = None
        return conn

    def copy(self):
        # ``DatabaseBackend.copy`` would re-enter ``__init__`` as
        # (uri, migration_table) and mis-assign the migration table to the
        # dbname; keep the same dbname instead so copies share this database.
        return _NamedMemoryBackend(self.uri, self._dbname, self.migration_table)

class _FileBackend(_FastLogMixin, SQLiteBackend):
    """File-backed SQLite backend with the fast (non-DNS) migration log.

    Mirrors what ``yoyo.get_backend`` returns (parsed URI + ``init_database``),
    so the file-backed storage behaves identically to the original.
    """

    def __init__(self, dburi, migration_table: str = "_yoyo_migration") -> None:
        super().__init__(dburi, migration_table)
        self.init_database()


# Shared SELECT for users. Roles live in the many-to-many user_roles table, so
# this flattens them into a comma-separated string via a correlated subquery;
# ``_row_to_user`` splits it back into ``User.roles`` (sorted for determinism).
_USER_SELECT = """
SELECT u.id, u.username, u.email, u.password_hash, u.created_at,
       u.email_verified, u.verification_token_hash, u.verification_expires_at,
       (SELECT GROUP_CONCAT(r.name, ',')
        FROM user_roles ur
        JOIN roles r ON r.id = ur.role_id
        WHERE ur.user_id = u.id) AS roles
FROM users u
"""


def _now() -> str:
    """Current UTC time as an ISO-8601 string (patchable in tests)."""
    return datetime.now(timezone.utc).isoformat()


def _parse_answer(raw: str) -> str | int | float:
    """Best-effort coercion of a stored answer back to its original type.

    ``correct_answer`` may be an int, float, or string; SQLite stores all of
    them as TEXT, so on read we try to recover the numeric form.
    """
    for cast in (int, float):
        try:
            return cast(raw)
        except (ValueError, TypeError):
            continue
    return raw


class SqliteStorage(Storage):
    """SQLite-backed :class:`Storage`, in-memory by default.

    The schema is defined by versioned SQL migrations under ``migrations/`` and
    applied through yoyo-migrations on construction (already-applied revisions
    are skipped, so this is idempotent). Pass a file path to persist across
    runs instead of the default ``":memory:"``.

    yoyo owns the connection: it applies each migration on an independent
    connection that shares the same (private named) in-memory database, and
    this class reuses yoyo's connection for all queries so the single DB stays
    coherent. Access is guarded by a re-entrant lock.

    Note: Python 3.14 removed the post-connect ``check_same_thread`` switch, so
    keep all access to one instance on the thread that constructed it (create a
    fresh storage per thread if the portal ever calls in from background
    threads).
    """

    def __init__(
        self,
        path: str = ":memory:",
        migrations_dir: str | Path = MIGRATIONS_DIR,
    ) -> None:
        self._lock = threading.RLock()
        # Each in-memory storage gets a private named shared-cache database so
        # instances never share data; a file path persists across runs instead.
        backend = (
            _NamedMemoryBackend(
                parse_uri("sqlite:///:memory:"), f"agentic_portal_{uuid.uuid4().hex}"
            )
            if path == ":memory:"
            else _FileBackend(parse_uri(f"sqlite:///{path}"))
        )
        with self._lock:
            self._conn = backend.connection
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            # yoyo's migration bookkeeping binds datetimes through a pre-3.12
            # adapter that Python now deprecates; that warning is yoyo-internal,
            # so don't surface it on every storage construction.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The default datetime adapter is deprecated",
                    category=DeprecationWarning,
                )
                backend.create_lock_table()
                backend.ensure_internal_schema_updated()
                migrations = read_migrations(str(migrations_dir))
                with backend.lock():
                    backend.apply_migrations(backend.to_apply(migrations))
            # After the schema is in place, provision the bootstrap admin
            # (admin + teacher) from .env when the vars are set.
            self.seed_admin_from_env()

    def close(self) -> None:
        """Close the underlying database connection (dropping a memory DB)."""
        with self._lock:
            self._conn.close()

    # --- internal helpers ------------------------------------------------------

    def _role_id(self, name: RoleName) -> int:
        row = self._conn.execute(
            "SELECT id FROM roles WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown role: {name}")
        return row["id"]

    def _row_to_user(self, row: sqlite3.Row) -> User:
        roles = sorted(row["roles"].split(",")) if row["roles"] else []
        return User(
            id=row["id"],
            username=row["username"],
            roles=roles,
            email=row["email"],
            email_verified=bool(row["email_verified"]),
            password_hash=row["password_hash"],
            verification_token_hash=row["verification_token_hash"],
            verification_expires_at=row["verification_expires_at"],
            created_at=row["created_at"],
        )

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"],
            topic=row["topic"],
            text=row["text"],
            complexity=row["complexity"],
            correct_answer=_parse_answer(row["correct_answer"]),
            solution=row["solution"],
        )

    # --- roles ---------------------------------------------------------------

    def list_roles(self) -> list[Role]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name FROM roles ORDER BY id"
            ).fetchall()
        return [Role(id=r["id"], name=r["name"]) for r in rows]

    def get_role(self, name: RoleName) -> Role | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name FROM roles WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            return None
        return Role(id=row["id"], name=row["name"])

    # --- users ---------------------------------------------------------------

    def create_user(
        self,
        username: str,
        roles: RoleName | Sequence[RoleName],
        *,
        email: str | None = None,
        password: str | None = None,
        email_verified: bool = True,
    ) -> User:
        role_names = [roles] if isinstance(roles, str) else list(roles)
        role_names = sorted(set(role_names))
        with self._lock:
            # Resolve every role before inserting so an unknown role leaves no
            # partially-created user behind.
            role_ids = [self._role_id(r) for r in role_names]
            password_hash = hash_password(password) if password else None
            cursor = self._conn.execute(
                "INSERT INTO users (username, email, password_hash, email_verified,"
                " created_at) VALUES (?, ?, ?, ?, ?)",
                (username, email, password_hash, int(email_verified), _now()),
            )
            user_id = cursor.lastrowid
            for role_id in role_ids:
                self._conn.execute(
                    "INSERT INTO user_roles (user_id, role_id) VALUES (?, ?)",
                    (user_id, role_id),
                )
            self._conn.commit()
            return self.get_user(user_id)

    def get_user(self, user_id: int) -> User | None:
        with self._lock:
            row = self._conn.execute(
                _USER_SELECT + " WHERE u.id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_user(row)

    def get_user_by_username(self, username: str) -> User | None:
        with self._lock:
            row = self._conn.execute(
                _USER_SELECT + " WHERE u.username = ?", (username,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_user(row)

    def list_users(self, *, role: RoleName | None = None) -> list[User]:
        sql = _USER_SELECT
        params: tuple = ()
        if role is not None:
            sql += (
                " WHERE EXISTS (SELECT 1 FROM user_roles ur"
                " JOIN roles r ON r.id = ur.role_id"
                " WHERE ur.user_id = u.id AND r.name = ?)"
            )
            params = (role,)
        sql += " ORDER BY u.id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_user(r) for r in rows]

    def add_role(self, user_id: int, role: RoleName) -> User:
        with self._lock:
            if self.get_user(user_id) is None:
                raise ValueError(f"No user with id {user_id}")
            role_id = self._role_id(role)
            self._conn.execute(
                "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
                (user_id, role_id),
            )
            self._conn.commit()
            return self.get_user(user_id)

    def remove_role(self, user_id: int, role: RoleName) -> User:
        with self._lock:
            if self.get_user(user_id) is None:
                raise ValueError(f"No user with id {user_id}")
            role_id = self._role_id(role)
            self._conn.execute(
                "DELETE FROM user_roles WHERE user_id = ? AND role_id = ?",
                (user_id, role_id),
            )
            self._conn.commit()
            return self.get_user(user_id)

    def update_user(
        self,
        user_id: int,
        *,
        username: str | None = None,
        email: str | None = None,
        roles: Sequence[RoleName] | None = None,
    ) -> User:
        """Update the editable fields of an existing user and return it."""
        with self._lock:
            current = self.get_user(user_id)
            if current is None:
                raise ValueError(f"No user with id {user_id}")
            if username is not None:
                self._conn.execute(
                    "UPDATE users SET username = ? WHERE id = ?",
                    (username, user_id),
                )
            if email is not None and email != current.email:
                # A changed address is unverified again and any pending invite
                # token no longer belongs to the new address.
                self._conn.execute(
                    "UPDATE users SET email = ?, email_verified = 0,"
                    " verification_token_hash = NULL,"
                    " verification_expires_at = NULL WHERE id = ?",
                    (email, user_id),
                )
            if roles is not None:
                role_ids = [self._role_id(r) for r in sorted(set(roles))]
                self._conn.execute(
                    "DELETE FROM user_roles WHERE user_id = ?", (user_id,)
                )
                for role_id in role_ids:
                    self._conn.execute(
                        "INSERT INTO user_roles (user_id, role_id) VALUES (?, ?)",
                        (user_id, role_id),
                    )
            self._conn.commit()
            return self.get_user(user_id)

    def issue_verification_token(
        self,
        user_id: int,
        *,
        ttl_days: int = 7,
    ) -> str:
        """Generate and store a verification token for ``user_id``."""
        with self._lock:
            if self.get_user(user_id) is None:
                raise ValueError(f"No user with id {user_id}")
            raw = secrets.token_urlsafe(32)
            expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).isoformat()
            self._conn.execute(
                "UPDATE users SET verification_token_hash = ?,"
                " verification_expires_at = ? WHERE id = ?",
                (hash_token(raw), expires_at, user_id),
            )
            self._conn.commit()
            return raw

    def get_user_by_verification_token(self, token: str) -> User | None:
        """Return the user whose pending token hashes to ``token``."""
        with self._lock:
            row = self._conn.execute(
                _USER_SELECT
                + " WHERE u.verification_token_hash = ? AND u.email_verified = 0",
                (hash_token(token),),
            ).fetchone()
        if row is None:
            return None
        user = self._row_to_user(row)
        if user.verification_expires_at is None:
            return None
        try:
            expires_at = datetime.fromisoformat(user.verification_expires_at)
        except ValueError:
            return None
        if expires_at <= datetime.now(timezone.utc):
            return None
        return user

    def complete_email_verification(self, user_id: int) -> User:
        """Mark the user's email verified and clear its pending token."""
        with self._lock:
            if self.get_user(user_id) is None:
                raise ValueError(f"No user with id {user_id}")
            self._conn.execute(
                "UPDATE users SET email_verified = 1,"
                " verification_token_hash = NULL,"
                " verification_expires_at = NULL WHERE id = ?",
                (user_id,),
            )
            self._conn.commit()
            return self.get_user(user_id)

    # --- passwords & credentials ---------------------------------------------

    def set_password(self, user_id: int, password: str) -> User:
        """Set (or reset) the user's password hash and return the user."""
        with self._lock:
            if self.get_user(user_id) is None:
                raise ValueError(f"No user with id {user_id}")
            self._conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (hash_password(password), user_id),
            )
            self._conn.commit()
            return self.get_user(user_id)

    def verify_credentials(self, username: str, password: str) -> User | None:
        """Return the user if ``password`` matches their stored hash, else ``None``.

        An unverified user never matches, so an invited account can't sign in
        before opening the verification link.
        """
        user = self.get_user_by_username(username)
        if user is None or not user.email_verified or not user.password_hash:
            return None
        if verify_password(password, user.password_hash):
            return user
        return None

    def seed_admin_from_env(self) -> User | None:
        """Create the initial admin (admin + teacher) from ``ADMIN_USERNAME`` /
        ``ADMIN_PASSWORD`` env vars.

        Idempotent: skipped entirely when either var is unset. When a user with
        that username already exists it is left untouched — except that a missing
        hash is backfilled, so an admin that predates the password column can
        still log in. Never overwrites an existing hash (a manually-changed
        password survives re-runs). Returns the affected user, or ``None`` when
        there was nothing to do. Callers must have loaded ``.env`` (the portal
        entry points already do) so the seed picks up local development
        credentials.
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
            cursor = self._conn.execute(
                "INSERT INTO tasks (topic, text, complexity, correct_answer, solution)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    task.topic,
                    task.text,
                    task.complexity,
                    str(task.correct_answer),
                    task.solution,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            self._conn.commit()
        return self._row_to_task(row)

    def get_task(self, task_id: int) -> Task | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def list_tasks(self) -> list[Task]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks ORDER BY id"
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def update_task(
        self,
        task_id: int,
        *,
        topic: str | None = None,
        text: str | None = None,
        complexity: str | None = None,
        correct_answer: str | int | float | None = None,
        solution: str | None = None,
    ) -> Task:
        """Update the editable fields of an existing task and return it.

        Only the fields given are changed; ``None`` leaves a field untouched.
        Raises ``ValueError`` when no task with ``task_id`` exists.
        """
        with self._lock:
            current = self._conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if current is None:
                raise ValueError(f"No task with id {task_id}")
            self._conn.execute(
                "UPDATE tasks SET topic = ?, text = ?, complexity = ?,"
                " correct_answer = ?, solution = ? WHERE id = ?",
                (
                    current["topic"] if topic is None else topic,
                    current["text"] if text is None else text,
                    current["complexity"] if complexity is None else complexity,
                    current["correct_answer"]
                    if correct_answer is None else str(correct_answer),
                    current["solution"] if solution is None else solution,
                    task_id,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
            self._conn.commit()
        return self._row_to_task(row)

    # --- assignments ---------------------------------------------------------

    def create_assignment(
        self,
        title: str,
        created_by: int,
        *,
        assigned_to: int | None = None,
    ) -> Assignment:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO assignments (title, created_by, assigned_to, created_at)"
                " VALUES (?, ?, ?, ?)",
                (title, created_by, assigned_to, _now()),
            )
            row = self._conn.execute(
                "SELECT * FROM assignments WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            self._conn.commit()
        return Assignment(
            **dict(row),
            tasks=LazyTaskList(
                0, lambda aid=row["id"]: self.list_assignment_tasks(aid)
            ),
        )

    def get_assignment(self, assignment_id: int) -> Assignment | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT a.*,"
                " (SELECT COUNT(*) FROM assignment_tasks at"
                "  WHERE at.assignment_id = a.id) AS task_count"
                " FROM assignments a WHERE a.id = ?",
                (assignment_id,),
            ).fetchone()
        if row is None:
            return None
        return Assignment(
            **dict(row),
            tasks=LazyTaskList(
                row["task_count"], lambda aid=row["id"]: self.list_assignment_tasks(aid)
            ),
        )

    def list_assignments(
        self,
        *,
        created_by: int | None = None,
        assigned_to: int | None = None,
    ) -> list[Assignment]:
        sql = (
            "SELECT a.*,"
            " (SELECT COUNT(*) FROM assignment_tasks at"
            "  WHERE at.assignment_id = a.id) AS task_count"
            " FROM assignments a"
        )
        clauses: list[str] = []
        params: list[object] = []
        if created_by is not None:
            clauses.append("a.created_by = ?")
            params.append(created_by)
        if assigned_to is not None:
            clauses.append("a.assigned_to = ?")
            params.append(assigned_to)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY a.id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            Assignment(
                **dict(r),
                tasks=LazyTaskList(
                    r["task_count"], lambda aid=r["id"]: self.list_assignment_tasks(aid)
                ),
            )
            for r in rows
        ]

    def delete_assignment(self, assignment_id: int) -> None:
        """Delete an assignment, its link rows, and the tasks that belonged only to it.

        A task still referenced by another assignment survives the cascade.
        Raises ``ValueError`` when no assignment with ``assignment_id`` exists.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM assignments WHERE id = ?", (assignment_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"No assignment with id {assignment_id}")

            task_ids = [
                r["task_id"]
                for r in self._conn.execute(
                    "SELECT task_id FROM assignment_tasks WHERE assignment_id = ?",
                    (assignment_id,),
                ).fetchall()
            ]

            # Drop this assignment's links first so the task delete below never
            # trips on its own link rows (works with RESTRICT or CASCADE FKs).
            self._conn.execute(
                "DELETE FROM assignment_tasks WHERE assignment_id = ?",
                (assignment_id,),
            )

            # Delete the tasks that were exclusively this assignment's; a task
            # still referenced by another assignment must survive.
            if task_ids:
                placeholders = ", ".join("?" * len(task_ids))
                self._conn.execute(
                    f"DELETE FROM tasks WHERE id IN ({placeholders})"
                    " AND NOT EXISTS (SELECT 1 FROM assignment_tasks at2"
                    " WHERE at2.task_id = tasks.id)",
                    task_ids,
                )

            self._conn.execute(
                "DELETE FROM assignments WHERE id = ?", (assignment_id,)
            )
            self._conn.commit()

    def add_task_to_assignment(
        self,
        assignment_id: int,
        task_id: int,
        *,
        position: int | None = None,
    ) -> None:
        with self._lock:
            if position is None:
                row = self._conn.execute(
                    "SELECT COALESCE(MAX(position), -1) AS pos"
                    " FROM assignment_tasks WHERE assignment_id = ?",
                    (assignment_id,),
                ).fetchone()
                position = row["pos"] + 1
            self._conn.execute(
                "INSERT INTO assignment_tasks (assignment_id, task_id, position)"
                " VALUES (?, ?, ?)",
                (assignment_id, task_id, position),
            )
            self._conn.commit()

    def assign_assignment(
            self,
            assignment_id: int,
            assign_to: int | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE assignments SET assigned_to = ? WHERE id = ?",
                (assign_to, assignment_id),
            )
            self._conn.commit()

    def grant_extra_attempt(self, assignment_id: int) -> Assignment:
        with self._lock:
            self._conn.execute(
                "UPDATE assignments SET extra_attempts = extra_attempts + 1"
                " WHERE id = ?",
                (assignment_id,),
            )
            self._conn.commit()
        assignment = self.get_assignment(assignment_id)
        if assignment is None:
            raise ValueError(f"No assignment with id {assignment_id}")
        return assignment

    def list_assignment_tasks(self, assignment_id: int) -> list[Task]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT t.* FROM assignment_tasks at"
                " JOIN tasks t ON t.id = at.task_id"
                " WHERE at.assignment_id = ?"
                " ORDER BY at.position, t.id",
                (assignment_id,),
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def remove_task_from_assignment(self, assignment_id: int, task_id: int) -> None:
        """Unlink ``task_id`` from ``assignment_id``'s task list.

        The task is deleted when no other assignment references it (mirroring
        ``delete_assignment``'s shared-task semantics). Raises ``ValueError``
        when the assignment is missing or the task is not part of it.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM assignments WHERE id = ?", (assignment_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"No assignment with id {assignment_id}")

            link = self._conn.execute(
                "SELECT task_id FROM assignment_tasks"
                " WHERE assignment_id = ? AND task_id = ?",
                (assignment_id, task_id),
            ).fetchone()
            if link is None:
                raise ValueError(f"Task {task_id} not in assignment {assignment_id}")

            self._conn.execute(
                "DELETE FROM assignment_tasks WHERE assignment_id = ? AND task_id = ?",
                (assignment_id, task_id),
            )
            # Delete the task only when no other assignment references it.
            self._conn.execute(
                "DELETE FROM tasks WHERE id = ? AND NOT EXISTS"
                " (SELECT 1 FROM assignment_tasks at2 WHERE at2.task_id = tasks.id)",
                (task_id,),
            )
            self._conn.commit()

    # --- attempts ------------------------------------------------------------

    def start_attempt(self, assignment_id: int, student_id: int) -> Attempt:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO attempts (assignment_id, student_id, status, started_at)"
                " VALUES (?, ?, 'in_progress', ?)",
                (assignment_id, student_id, _now()),
            )
            row = self._conn.execute(
                "SELECT * FROM attempts WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            self._conn.commit()
        return Attempt(**dict(row))

    def complete_attempt(self, attempt_id: int) -> Attempt:
        with self._lock:
            self._conn.execute(
                "UPDATE attempts SET status = 'completed', completed_at = ?"
                " WHERE id = ?",
                (_now(), attempt_id),
            )
            row = self._conn.execute(
                "SELECT * FROM attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            self._conn.commit()
        if row is None:
            raise ValueError(f"No attempt with id {attempt_id}")
        return Attempt(**dict(row))

    def get_attempt(self, attempt_id: int) -> Attempt | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
        if row is None:
            return None
        return Attempt(**dict(row))

    def list_attempts(
        self,
        *,
        student_id: int | None = None,
        assignment_id: int | None = None,
        results_seen: bool | None = None,
    ) -> list[Attempt]:
        sql = "SELECT * FROM attempts"
        clauses: list[str] = []
        params: list[object] = []
        if student_id is not None:
            clauses.append("student_id = ?")
            params.append(student_id)
        if assignment_id is not None:
            clauses.append("assignment_id = ?")
            params.append(assignment_id)
        if results_seen is not None:
            clauses.append("results_seen = ?")
            params.append(int(results_seen))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [Attempt(**dict(r)) for r in rows]

    def mark_results_seen(self, attempt_id: int) -> Attempt:
        with self._lock:
            self._conn.execute(
                "UPDATE attempts SET results_seen = 1 WHERE id = ?", (attempt_id,)
            )
            row = self._conn.execute(
                "SELECT * FROM attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            self._conn.commit()
        if row is None:
            raise ValueError(f"No attempt with id {attempt_id}")
        return Attempt(**dict(row))

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
        # Upsert on (attempt_id, task_id) — saving progress and later grading go
        # through the same row, so a re-save overwrites instead of appending.
        # The row is re-read by key rather than by ``lastrowid``, which is not
        # meaningful on the DO UPDATE branch.
        with self._lock:
            self._conn.execute(
                "INSERT INTO attempt_results"
                " (attempt_id, task_id, given_answer, expected_answer, is_correct,"
                "  score, detail)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(attempt_id, task_id) DO UPDATE SET"
                "  given_answer = excluded.given_answer,"
                "  expected_answer = excluded.expected_answer,"
                "  is_correct = excluded.is_correct,"
                "  score = excluded.score,"
                "  detail = excluded.detail",
                (
                    attempt_id,
                    task_id,
                    given_answer,
                    expected_answer,
                    int(is_correct) if is_correct is not None else None,
                    score,
                    detail,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM attempt_results WHERE attempt_id = ? AND task_id = ?",
                (attempt_id, task_id),
            ).fetchone()
            self._conn.commit()
        return self._row_to_result(row)

    def list_results(
        self,
        *,
        attempt_id: int | None = None,
        student_id: int | None = None,
    ) -> list[AttemptResult]:
        sql = "SELECT r.* FROM attempt_results r"
        clauses: list[str] = []
        params: list[object] = []
        if attempt_id is not None:
            clauses.append("r.attempt_id = ?")
            params.append(attempt_id)
        if student_id is not None:
            sql += " JOIN attempts a ON a.id = r.attempt_id"
            clauses.append("a.student_id = ?")
            params.append(student_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY r.id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_result(r) for r in rows]

    def _row_to_result(self, row: sqlite3.Row) -> AttemptResult:
        return AttemptResult(
            id=row["id"],
            attempt_id=row["attempt_id"],
            task_id=row["task_id"],
            given_answer=row["given_answer"],
            expected_answer=row["expected_answer"],
            is_correct=(
                bool(row["is_correct"]) if row["is_correct"] is not None else None
            ),
            score=row["score"],
            detail=row["detail"],
        )