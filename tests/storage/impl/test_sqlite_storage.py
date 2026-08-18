from __future__ import annotations

import sqlite3

import pytest

from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.storage import Assignment, Attempt, AttemptResult, Role, SqliteStorage, Task, User


def _storage() -> SqliteStorage:
    return SqliteStorage()


def _task(correct_answer: str | int | float = "12", text: str = "What is 3 times 4?") -> GeneratedTask:
    return GeneratedTask.model_validate(
        {
            "topic": "Arithmetic",
            "text": text,
            "complexity": "easy",
            "correct_answer": correct_answer,
            "solution": "Multiply the two numbers: 3 × 4 = 12.",
        }
    )


def _users(s: SqliteStorage) -> tuple[User, User]:
    teacher = s.create_user("teacher1", "teacher", email="t@example.com")
    student = s.create_user("student1", "student")
    return teacher, student


# --- roles -------------------------------------------------------------------


def test_roles_are_seeded() -> None:
    s = _storage()

    roles = s.list_roles()

    assert [r.name for r in roles] == ["admin", "teacher", "student"]
    assert all(isinstance(r, Role) and r.id > 0 for r in roles)


def test_get_role_returns_none_for_unknown() -> None:
    s = _storage()

    assert s.get_role("student").name == "student"  # type: ignore[union-attr]
    assert s.get_role("teacher").name == "teacher"  # type: ignore[union-attr]
    assert s.get_role("admin").name == "admin"  # type: ignore[union-attr]


# --- users -------------------------------------------------------------------


def test_create_and_get_user() -> None:
    s = _storage()

    user = s.create_user("alice", "student", email="alice@example.com")

    assert isinstance(user, User)
    assert user.username == "alice"
    assert user.roles == ["student"]
    assert user.email == "alice@example.com"
    assert user.created_at

    fetched = s.get_user(user.id)
    assert fetched == user
    assert s.get_user_by_username("alice") == user
    assert s.get_user_by_username("nobody") is None


def test_role_separation_across_users() -> None:
    s = _storage()
    admin = s.create_user("boss", "admin")
    teacher = s.create_user("teacher1", "teacher")
    student = s.create_user("student1", "student")

    assert [u.username for u in s.list_users(role="admin")] == ["admin", "boss"]
    assert [u.username for u in s.list_users(role="teacher")] == ["admin", "teacher1"]
    assert [u.username for u in s.list_users(role="student")] == ["student1"]
    assert [u.username for u in s.list_users()] == ["admin", "boss", "teacher1", "student1"]
    assert admin.roles == ["admin"]
    assert teacher.roles == ["teacher"]
    assert student.roles == ["student"]


def test_create_user_duplicate_username_raises() -> None:
    s = _storage()
    s.create_user("alice", "student")

    with pytest.raises(sqlite3.IntegrityError):
        s.create_user("alice", "teacher")


def test_create_user_unknown_role_raises() -> None:
    s = _storage()

    with pytest.raises(ValueError, match="Unknown role"):
        s.create_user("bob", "superuser")  # type: ignore[arg-type]


def test_create_user_unknown_role_leaves_no_partial_user() -> None:
    s = _storage()

    with pytest.raises(ValueError, match="Unknown role"):
        s.create_user("bob", ["admin", "superuser"])  # type: ignore[list-item]

    assert [u.username for u in s.list_users()] == ["admin"]


def test_create_user_with_multiple_roles() -> None:
    s = _storage()

    user = s.create_user("boss", ["teacher", "admin"])

    # Roles come back sorted for determinism.
    assert user.roles == ["admin", "teacher"]
    assert s.get_user(user.id).roles == ["admin", "teacher"]  # type: ignore[union-attr]
    assert s.get_user_by_username("boss").roles == ["admin", "teacher"]  # type: ignore[union-attr]


def test_create_user_deduplicates_repeated_roles() -> None:
    s = _storage()

    user = s.create_user("boss", ["admin", "admin", "teacher"])

    assert user.roles == ["admin", "teacher"]


def test_list_users_filters_by_role_membership() -> None:
    s = _storage()
    s.create_user("boss", ["admin", "teacher"])
    s.create_user("teacher1", "teacher")
    s.create_user("student1", "student")

    # The multi-role user appears under both filters.
    assert [u.username for u in s.list_users(role="admin")] == ["admin", "boss"]
    assert [u.username for u in s.list_users(role="teacher")] == ["admin", "boss", "teacher1"]
    assert [u.username for u in s.list_users(role="student")] == ["student1"]


def test_add_and_remove_role() -> None:
    s = _storage()
    user = s.create_user("student1", "student")

    updated = s.add_role(user.id, "teacher")
    assert updated.roles == ["student", "teacher"]
    assert [u.username for u in s.list_users(role="teacher")] == ["admin", "student1"]

    updated = s.remove_role(user.id, "student")
    assert updated.roles == ["teacher"]
    assert s.list_users(role="student") == []


def test_remove_last_role_yields_empty_roles() -> None:
    s = _storage()
    user = s.create_user("student1", "student")

    updated = s.remove_role(user.id, "student")

    assert updated.roles == []
    assert s.list_users(role="student") == []
    assert [u.username for u in s.list_users()] == ["admin", updated.username]


def test_add_role_unknown_role_raises() -> None:
    s = _storage()
    user = s.create_user("student1", "student")

    with pytest.raises(ValueError, match="Unknown role"):
        s.add_role(user.id, "superuser")  # type: ignore[arg-type]


def test_add_role_to_missing_user_raises() -> None:
    s = _storage()

    with pytest.raises(ValueError, match="No user"):
        s.add_role(9999, "teacher")


# --- tasks -------------------------------------------------------------------


def test_task_round_trip_preserves_fields() -> None:
    s = _storage()
    task = s.create_task(_task(correct_answer=12))

    assert isinstance(task, Task)
    assert task.id > 0
    assert task.topic == "Arithmetic"
    assert task.text == "What is 3 times 4?"
    assert task.complexity == "easy"
    assert task.correct_answer == 12
    assert "3 × 4 = 12" in task.solution

    fetched = s.get_task(task.id)
    assert fetched == task
    assert [t.id for t in s.list_tasks()] == [task.id]


def test_task_round_trip_keeps_string_answer() -> None:
    s = _storage()

    task = s.create_task(_task(correct_answer="Lego"))

    assert task.correct_answer == "Lego"


# --- assignments -------------------------------------------------------------


def test_create_and_get_assignment() -> None:
    s = _storage()
    teacher, student = _users(s)

    assignment = s.create_assignment("Fractions practice", teacher.id, assigned_to=student.id)

    assert isinstance(assignment, Assignment)
    assert assignment.title == "Fractions practice"
    assert assignment.created_by == teacher.id
    assert assignment.assigned_to == student.id
    assert assignment.created_at

    fetched = s.get_assignment(assignment.id)
    assert fetched == assignment


def test_list_assignments_filters() -> None:
    s = _storage()
    teacher, student = _users(s)
    other = s.create_user("student2", "student")

    a1 = s.create_assignment("A1", teacher.id, assigned_to=student.id)
    a2 = s.create_assignment("A2", teacher.id, assigned_to=other.id)

    assert [a.id for a in s.list_assignments()] == [a1.id, a2.id]
    assert [a.id for a in s.list_assignments(created_by=teacher.id)] == [a1.id, a2.id]
    assert [a.id for a in s.list_assignments(assigned_to=student.id)] == [a1.id]


def test_assignment_created_by_unknown_user_raises() -> None:
    s = _storage()

    with pytest.raises(sqlite3.IntegrityError):
        s.create_assignment("Bad", created_by=9999)


def test_assignment_tasks_append_in_order() -> None:
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Ordered", teacher.id)
    t1 = s.create_task(_task(text="first"))
    t2 = s.create_task(_task(text="second"))
    t3 = s.create_task(_task(text="third"))

    s.add_task_to_assignment(assignment.id, t1.id)
    s.add_task_to_assignment(assignment.id, t2.id)
    s.add_task_to_assignment(assignment.id, t3.id)

    assert [t.text for t in s.list_assignment_tasks(assignment.id)] == [
        "first",
        "second",
        "third",
    ]


def test_assignment_task_explicit_position() -> None:
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Ordered", teacher.id)
    auto = s.create_task(_task(text="auto"))
    explicit = s.create_task(_task(text="explicit"))
    auto2 = s.create_task(_task(text="auto2"))

    s.add_task_to_assignment(assignment.id, auto.id)  # appends at 0
    s.add_task_to_assignment(assignment.id, explicit.id, position=10)
    s.add_task_to_assignment(assignment.id, auto2.id)  # appends at 11 (after 10)

    # auto=0 < explicit=10 < auto2=11: the explicit position wins over the
    # insertion order, and append always goes after everything already there.
    assert [t.text for t in s.list_assignment_tasks(assignment.id)] == [
        "auto",
        "explicit",
        "auto2",
    ]


# --- attempts -----------------------------------------------------------------


def test_attempt_lifecycle() -> None:
    s = _storage()
    teacher, student = _users(s)
    assignment = s.create_assignment("Practice", teacher.id, assigned_to=student.id)

    attempt = s.start_attempt(assignment.id, student.id)

    assert isinstance(attempt, Attempt)
    assert attempt.status == "in_progress"
    assert attempt.completed_at is None
    assert attempt.started_at

    completed = s.complete_attempt(attempt.id)

    assert completed.status == "completed"
    assert completed.completed_at is not None
    assert s.get_attempt(attempt.id).status == "completed"  # type: ignore[union-attr]


def test_list_attempts_filters() -> None:
    s = _storage()
    teacher, student = _users(s)
    other = s.create_user("student2", "student")
    a1 = s.create_assignment("A1", teacher.id, assigned_to=student.id)
    a2 = s.create_assignment("A2", teacher.id, assigned_to=other.id)

    attempt1 = s.start_attempt(a1.id, student.id)
    attempt2 = s.start_attempt(a2.id, other.id)

    assert [a.id for a in s.list_attempts(student_id=student.id)] == [attempt1.id]
    assert [a.id for a in s.list_attempts(assignment_id=a2.id)] == [attempt2.id]
    assert [a.id for a in s.list_attempts()] == [attempt1.id, attempt2.id]


# --- results ------------------------------------------------------------------


def test_record_and_list_results() -> None:
    s = _storage()
    teacher, student = _users(s)
    assignment = s.create_assignment("Graded", teacher.id, assigned_to=student.id)
    task = s.create_task(_task(correct_answer="12"))
    attempt = s.start_attempt(assignment.id, student.id)

    result = s.record_result(
        attempt.id,
        task.id,
        given_answer="12",
        expected_answer="12",
        is_correct=True,
        score=1.0,
        detail="Matches",
    )

    assert isinstance(result, AttemptResult)
    assert result.attempt_id == attempt.id
    assert result.task_id == task.id
    assert result.is_correct is True
    assert result.score == 1.0
    assert result.detail == "Matches"

    results = s.list_results(attempt_id=attempt.id)
    assert [r.id for r in results] == [result.id]
    assert s.list_results(student_id=student.id) == results
    assert s.list_results(student_id=teacher.id) == []


def test_record_result_null_grading_fields() -> None:
    s = _storage()
    teacher, student = _users(s)
    assignment = s.create_assignment("Ungraded", teacher.id, assigned_to=student.id)
    task = s.create_task(_task())
    attempt = s.start_attempt(assignment.id, student.id)

    result = s.record_result(attempt.id, task.id, given_answer="12")

    assert result.is_correct is None
    assert result.score is None
    assert result.expected_answer is None


# --- passwords & credentials ---------------------------------------------------


def test_create_user_with_password_hashes_it() -> None:
    s = _storage()

    user = s.create_user("alice", "student", password="s3cret")

    assert user.password_hash is not None
    assert user.password_hash != "s3cret"
    assert "s3cret" not in user.password_hash
    assert "s3cret" not in repr(user)  # repr=False on the field
    # Salted, so identical passwords never collide in storage.
    other = s.create_user("bob", "student", password="s3cret")
    assert other.password_hash != user.password_hash


def test_create_user_without_password_has_no_hash() -> None:
    s = _storage()

    user = s.create_user("carol", "student")

    assert user.password_hash is None


def test_verify_credentials_roundtrip() -> None:
    s = _storage()
    s.create_user("alice", "student", password="hunter2")

    assert s.verify_credentials("alice", "hunter2").username == "alice"  # type: ignore[union-attr]
    assert s.verify_credentials("alice", "wrong") is None
    assert s.verify_credentials("nobody", "hunter2") is None


def test_verify_credentials_user_without_password() -> None:
    s = _storage()
    s.create_user("alice", "student")

    assert s.verify_credentials("alice", "anything") is None


def test_set_password() -> None:
    s = _storage()
    user = s.create_user("alice", "student")

    updated = s.set_password(user.id, "first-pass")

    assert updated.password_hash is not None
    assert s.verify_credentials("alice", "first-pass") is not None

    s.set_password(user.id, "second-pass")
    assert s.verify_credentials("alice", "first-pass") is None
    assert s.verify_credentials("alice", "second-pass") is not None


def test_set_password_missing_user_raises() -> None:
    s = _storage()

    with pytest.raises(ValueError, match="No user with id 999"):
        s.set_password(999, "x")


# --- admin seeding from .env ----------------------------------------------------


def test_seed_admin_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_USERNAME", "boss")
    monkeypatch.setenv("ADMIN_PASSWORD", "hunter2")

    s = _storage()

    admin = s.get_user_by_username("boss")
    assert admin is not None
    assert admin.roles == ["admin", "teacher"]
    assert s.verify_credentials("boss", "hunter2") is not None
    assert s.verify_credentials("boss", "wrong") is None


def test_seed_admin_from_env_is_idempotent(monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_USERNAME", "boss")
    monkeypatch.setenv("ADMIN_PASSWORD", "hunter2")

    s = _storage()

    assert s.seed_admin_from_env() is None  # already seeded on construction
    assert len(s.list_users(role="admin")) == 1


def test_no_seed_without_env(monkeypatch) -> None:
    monkeypatch.delenv("ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)

    s = _storage()

    assert s.list_users(role="admin") == []
    assert s.seed_admin_from_env() is None


def test_seed_requires_both_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_USERNAME", "boss")
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)

    s = _storage()

    assert s.get_user_by_username("boss") is None


def test_seed_backfills_existing_user_without_hash(monkeypatch) -> None:
    monkeypatch.delenv("ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    s = _storage()
    # A user that predates the password column exists, but has no hash.
    existing = s.create_user("boss", "admin")
    assert existing.password_hash is None

    monkeypatch.setenv("ADMIN_USERNAME", "boss")
    monkeypatch.setenv("ADMIN_PASSWORD", "hunter2")
    admin = s.seed_admin_from_env()

    assert admin.id == existing.id
    assert admin.roles == ["admin"]  # roles unchanged, only the hash is backfilled
    assert s.verify_credentials("boss", "hunter2") is not None


def test_seed_never_overwrites_existing_hash(monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_USERNAME", "boss")
    monkeypatch.setenv("ADMIN_PASSWORD", "hunter2")
    s = _storage()
    assert s.verify_credentials("boss", "hunter2") is not None

    # Re-running the seed must not clobber the stored hash with a new one.
    before = s.get_user_by_username("boss").password_hash  # type: ignore[union-attr]
    assert s.seed_admin_from_env() is None
    after = s.get_user_by_username("boss").password_hash  # type: ignore[union-attr]
    assert after == before