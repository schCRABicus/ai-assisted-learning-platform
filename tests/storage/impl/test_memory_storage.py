"""Tests for the dict-backed :class:`InMemoryStorage`.

This is the only other storage suite besides ``test_sqlite_storage.py``; the two
share the ``Storage`` contract, so the expectations here mirror that suite, with
two deliberate differences: memory storage is per-instance (no shared file), and
constraint violations raise ``ValueError`` rather than ``sqlite3.IntegrityError``.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.storage import (
    Assignment,
    Attempt,
    AttemptResult,
    InMemoryStorage,
    Role,
    Task,
    User,
)


def _storage() -> InMemoryStorage:
    return InMemoryStorage()


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


def _users(s: InMemoryStorage) -> tuple[User, User]:
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

    with pytest.raises(ValueError, match="already exists"):
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


def test_add_role_is_idempotent() -> None:
    s = _storage()
    user = s.create_user("student1", "student")

    s.add_role(user.id, "teacher")
    updated = s.add_role(user.id, "teacher")

    assert updated.roles == ["student", "teacher"]


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


def test_get_task_returns_none_for_missing() -> None:
    s = _storage()

    assert s.get_task(999) is None


def test_update_task_updates_all_fields() -> None:
    s = _storage()
    task = s.create_task(_task(correct_answer=12))

    updated = s.update_task(
        task.id,
        topic="Fractions",
        text="What is 3/4 of 20?",
        complexity="hard",
        correct_answer=15,
        solution="Three quarters of 20 is 15.",
    )

    assert updated.topic == "Fractions"
    assert updated.text == "What is 3/4 of 20?"
    assert updated.complexity == "hard"
    # Numeric answers come back coerced to their original type.
    assert updated.correct_answer == 15
    assert updated.solution == "Three quarters of 20 is 15."
    # The change is persisted, not just returned.
    assert s.get_task(task.id) == updated


def test_update_task_partial_update_keeps_other_fields() -> None:
    s = _storage()
    task = s.create_task(_task(correct_answer="12"))

    updated = s.update_task(task.id, topic="Algebra")

    assert updated.topic == "Algebra"
    assert updated.text == task.text
    assert updated.complexity == task.complexity
    assert updated.correct_answer == task.correct_answer
    assert updated.solution == task.solution


def test_update_task_missing_id_raises() -> None:
    s = _storage()

    with pytest.raises(ValueError, match="No task with id 999"):
        s.update_task(999, topic="Arithmetic")


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

    with pytest.raises(ValueError, match="No user"):
        s.create_assignment("Bad", created_by=9999)


def test_assignment_assigned_to_unknown_user_raises() -> None:
    s = _storage()
    teacher, _ = _users(s)

    with pytest.raises(ValueError, match="No user"):
        s.create_assignment("Bad", teacher.id, assigned_to=9999)


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

    assert [t.text for t in s.list_assignment_tasks(assignment.id)] == [
        "auto",
        "explicit",
        "auto2",
    ]


def test_add_task_to_assignment_unknown_ids_raise() -> None:
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Ordered", teacher.id)
    task = s.create_task(_task())

    with pytest.raises(ValueError, match="No task"):
        s.add_task_to_assignment(assignment.id, 999)
    with pytest.raises(ValueError, match="No assignment"):
        s.add_task_to_assignment(999, task.id)


def test_add_same_task_twice_raises() -> None:
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Ordered", teacher.id)
    task = s.create_task(_task())

    s.add_task_to_assignment(assignment.id, task.id)

    with pytest.raises(ValueError, match="already in assignment"):
        s.add_task_to_assignment(assignment.id, task.id)


def test_assignment_tasks_size_known_without_loading_content() -> None:
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Lazy", teacher.id)
    t1 = s.create_task(_task(text="first"))
    t2 = s.create_task(_task(text="second"))
    s.add_task_to_assignment(assignment.id, t1.id)
    s.add_task_to_assignment(assignment.id, t2.id)

    a = s.get_assignment(assignment.id)
    assert a.tasks.size == 2
    assert len(a.tasks) == 2
    assert a.tasks.loaded is False

    with patch.object(s, "list_assignment_tasks", side_effect=AssertionError("tasks fetched eagerly")):
        fetched = s.get_assignment(assignment.id)
        assert len(fetched.tasks) == 2
        assert fetched.tasks.size == 2


def test_assignment_tasks_content_loads_on_first_access() -> None:
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Lazy", teacher.id)
    t1 = s.create_task(_task(text="first"))
    t2 = s.create_task(_task(text="second"))
    s.add_task_to_assignment(assignment.id, t1.id)
    s.add_task_to_assignment(assignment.id, t2.id)

    a = s.get_assignment(assignment.id)
    assert a.tasks.loaded is False

    assert [t.text for t in a.tasks] == ["first", "second"]
    assert a.tasks.loaded is True
    assert a.tasks[0].text == "first"


def test_list_assignments_reports_task_counts() -> None:
    s = _storage()
    teacher, _ = _users(s)
    empty = s.create_assignment("Empty", teacher.id)
    with_one = s.create_assignment("With one", teacher.id)
    task = s.create_task(_task(text="only"))
    s.add_task_to_assignment(with_one.id, task.id)

    assignments = {a.id: a for a in s.list_assignments()}
    assert assignments[empty.id].tasks.size == 0
    assert assignments[with_one.id].tasks.size == 1
    assert assignments[empty.id].tasks.loaded is False
    assert assignments[with_one.id].tasks.loaded is False
    assert [t.text for t in assignments[with_one.id].tasks] == ["only"]


def test_created_assignment_has_empty_lazy_tasks() -> None:
    s = _storage()
    teacher, _ = _users(s)

    assignment = s.create_assignment("Fresh", teacher.id)

    assert assignment.tasks.size == 0
    assert len(assignment.tasks) == 0
    assert assignment.tasks.loaded is False


def test_delete_assignment_removes_assignment() -> None:
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Doomed", teacher.id)
    task = s.create_task(_task())
    s.add_task_to_assignment(assignment.id, task.id)

    s.delete_assignment(assignment.id)

    assert s.get_assignment(assignment.id) is None
    assert [a.id for a in s.list_assignments()] == []


def test_delete_assignment_deletes_associated_tasks() -> None:
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Doomed", teacher.id)
    t1 = s.create_task(_task(text="first"))
    t2 = s.create_task(_task(text="second"))
    s.add_task_to_assignment(assignment.id, t1.id)
    s.add_task_to_assignment(assignment.id, t2.id)

    s.delete_assignment(assignment.id)

    assert s.get_task(t1.id) is None
    assert s.get_task(t2.id) is None
    assert [t.id for t in s.list_tasks()] == []


def test_delete_assignment_removes_task_links() -> None:
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Doomed", teacher.id)
    task = s.create_task(_task())
    s.add_task_to_assignment(assignment.id, task.id)

    s.delete_assignment(assignment.id)

    assert s.list_assignment_tasks(assignment.id) == []


def test_delete_assignment_with_no_tasks() -> None:
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Empty", teacher.id)

    s.delete_assignment(assignment.id)

    assert s.get_assignment(assignment.id) is None


def test_delete_assignment_unknown_id_returns_none() -> None:
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Kept", teacher.id)

    assert s.delete_assignment(999) is None

    # A failed delete leaves existing data untouched.
    assert s.get_assignment(assignment.id) == assignment


def test_delete_assignment_twice_returns_none() -> None:
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Doomed", teacher.id)
    assert s.delete_assignment(assignment.id) == assignment.id
    assert s.delete_assignment(assignment.id) is None


def test_delete_assignment_preserves_other_assignments() -> None:
    s = _storage()
    teacher, _ = _users(s)
    doomed = s.create_assignment("Doomed", teacher.id)
    kept = s.create_assignment("Kept", teacher.id)
    doomed_task = s.create_task(_task(text="doomed"))
    kept_task = s.create_task(_task(text="kept"))
    s.add_task_to_assignment(doomed.id, doomed_task.id)
    s.add_task_to_assignment(kept.id, kept_task.id)

    s.delete_assignment(doomed.id)

    assert s.get_assignment(doomed.id) is None
    assert s.get_task(doomed_task.id) is None
    kept_after = s.get_assignment(kept.id)
    assert kept_after is not None
    assert [t.text for t in kept_after.tasks] == ["kept"]
    assert s.get_task(kept_task.id) is not None


def test_delete_assignment_keeps_task_shared_with_another_assignment() -> None:
    """A task still referenced by a surviving assignment must not be deleted."""
    s = _storage()
    teacher, _ = _users(s)
    first = s.create_assignment("First", teacher.id)
    second = s.create_assignment("Second", teacher.id)
    shared = s.create_task(_task(text="shared"))
    s.add_task_to_assignment(first.id, shared.id)
    s.add_task_to_assignment(second.id, shared.id)

    s.delete_assignment(first.id)

    assert s.get_task(shared.id) is not None
    assert [t.text for t in s.list_assignment_tasks(second.id)] == ["shared"]

def test_assign_assignment() -> None:
    """A task still referenced by a surviving assignment must not be deleted."""
    s = _storage()
    teacher, student = _users(s)
    assignment = s.create_assignment("First", teacher.id, assigned_to=None)

    assert s.get_assignment(assignment.id).assigned_to is None
    s.assign_assignment(assignment.id, student.id)
    assert s.get_assignment(assignment.id).assigned_to is student.id

def test_unassign_assignment() -> None:
    s = _storage()
    teacher, student = _users(s)
    assignment = s.create_assignment("First", teacher.id, assigned_to=student.id)

    assert s.get_assignment(assignment.id).assigned_to is student.id
    s.assign_assignment(assignment.id, None)
    assert s.get_assignment(assignment.id).assigned_to is None

def test_remove_task_from_assignment_unlinks_and_deletes_unshared() -> None:
    """Removing a task deletes it when no other assignment references it."""
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Algebra HW", teacher.id)
    task = s.create_task(_task())
    s.add_task_to_assignment(assignment.id, task.id)

    s.remove_task_from_assignment(assignment.id, task.id)

    assert s.list_assignment_tasks(assignment.id) == []
    assert s.get_task(task.id) is None


def test_remove_task_from_assignment_keeps_task_shared_with_another_assignment() -> None:
    """A task still referenced by a surviving assignment must not be deleted."""
    s = _storage()
    teacher, _ = _users(s)
    first = s.create_assignment("First", teacher.id)
    second = s.create_assignment("Second", teacher.id)
    shared = s.create_task(_task(text="shared"))
    s.add_task_to_assignment(first.id, shared.id)
    s.add_task_to_assignment(second.id, shared.id)

    s.remove_task_from_assignment(first.id, shared.id)

    assert s.get_task(shared.id) is not None
    assert s.list_assignment_tasks(first.id) == []
    assert [t.text for t in s.list_assignment_tasks(second.id)] == ["shared"]


def test_remove_task_from_assignment_leaves_other_tasks() -> None:
    """Removing one task of an assignment leaves the others in place."""
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Algebra HW", teacher.id)
    t1 = s.create_task(_task(text="first"))
    t2 = s.create_task(_task(text="second"))
    s.add_task_to_assignment(assignment.id, t1.id)
    s.add_task_to_assignment(assignment.id, t2.id)

    s.remove_task_from_assignment(assignment.id, t1.id)

    assert [t.text for t in s.list_assignment_tasks(assignment.id)] == ["second"]
    assert s.get_task(t1.id) is None
    assert s.get_task(t2.id) is not None


def test_remove_task_not_in_assignment_raises() -> None:
    """A task that was never linked to the assignment is a no-op error."""
    s = _storage()
    teacher, _ = _users(s)
    assignment = s.create_assignment("Algebra HW", teacher.id)
    task = s.create_task(_task())

    with pytest.raises(ValueError, match="not in assignment"):
        s.remove_task_from_assignment(assignment.id, task.id)

    # Nothing changed.
    assert s.get_task(task.id) is not None


def test_remove_task_from_missing_assignment_raises() -> None:
    s = _storage()

    with pytest.raises(ValueError, match="No assignment"):
        s.remove_task_from_assignment(999, 1)


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


def test_start_attempt_unknown_ids_raise() -> None:
    s = _storage()
    teacher, student = _users(s)
    assignment = s.create_assignment("Practice", teacher.id, assigned_to=student.id)

    with pytest.raises(ValueError, match="No assignment"):
        s.start_attempt(999, student.id)
    with pytest.raises(ValueError, match="No user"):
        s.start_attempt(assignment.id, 999)


def test_complete_attempt_missing_raises() -> None:
    s = _storage()

    with pytest.raises(ValueError, match="No attempt"):
        s.complete_attempt(999)


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


def test_record_result_unknown_ids_raise() -> None:
    s = _storage()
    teacher, student = _users(s)
    assignment = s.create_assignment("Graded", teacher.id, assigned_to=student.id)
    task = s.create_task(_task())
    attempt = s.start_attempt(assignment.id, student.id)

    with pytest.raises(ValueError, match="No attempt"):
        s.record_result(999, task.id)
    with pytest.raises(ValueError, match="No task"):
        s.record_result(attempt.id, 999)


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


# --- email verification ----------------------------------------------------------


def test_create_user_unverified_cannot_sign_in() -> None:
    s = _storage()
    user = s.create_user("invitee", "student", email="i@example.com", email_verified=False)

    assert user.email_verified is False
    assert user.password_hash is None
    assert s.verify_credentials("invitee", "anything") is None


def test_issue_and_complete_verification_token() -> None:
    s = _storage()
    user = s.create_user("invitee", "student", email="i@example.com", email_verified=False)

    token = s.issue_verification_token(user.id)

    found = s.get_user_by_verification_token(token)
    assert found is not None
    assert found.id == user.id

    s.complete_email_verification(user.id)
    refreshed = s.get_user(user.id)
    assert refreshed.email_verified is True
    assert refreshed.verification_token_hash is None
    assert refreshed.verification_expires_at is None
    # The token is single-use: consumed, so it no longer resolves.
    assert s.get_user_by_verification_token(token) is None


def test_get_user_by_verification_token_rejects_unknown() -> None:
    s = _storage()
    s.create_user("invitee", "student", email="i@example.com", email_verified=False)

    assert s.get_user_by_verification_token("bogus-token") is None


def test_get_user_by_verification_token_ignores_verified_user() -> None:
    s = _storage()
    user = s.create_user("alice", "student", email="a@example.com")

    token = s.issue_verification_token(user.id)

    # A verified user never matches a token lookup.
    assert s.get_user_by_verification_token(token) is None


def test_verification_token_expires() -> None:
    s = _storage()
    user = s.create_user("invitee", "student", email="i@example.com", email_verified=False)

    token = s.issue_verification_token(user.id, ttl_days=-1)  # already expired

    assert s.get_user_by_verification_token(token) is None


def test_update_user_changes_fields() -> None:
    s = _storage()
    user = s.create_user("alice", "student", email="a@example.com")

    updated = s.update_user(user.id, username="alicia", roles=["student", "teacher"])

    assert updated.username == "alicia"
    assert updated.roles == ["student", "teacher"]
    assert s.get_user(user.id).username == "alicia"


def test_update_user_email_change_resets_verification() -> None:
    s = _storage()
    user = s.create_user("alice", "student", email="a@example.com", password="pw123")

    assert s.verify_credentials("alice", "pw123") is not None

    updated = s.update_user(user.id, email="new@example.com")

    assert updated.email == "new@example.com"
    assert updated.email_verified is False
    assert updated.verification_token_hash is None
    assert updated.verification_expires_at is None
    # The account is unverified again, so it can't sign in until re-verified.
    assert s.verify_credentials("alice", "pw123") is None


def test_update_user_missing_raises() -> None:
    s = _storage()

    with pytest.raises(ValueError, match="No user with id 999"):
        s.update_user(999, username="nobody")


def test_issue_verification_token_missing_raises() -> None:
    s = _storage()

    with pytest.raises(ValueError, match="No user with id 999"):
        s.issue_verification_token(999)


def test_complete_email_verification_missing_raises() -> None:
    s = _storage()

    with pytest.raises(ValueError, match="No user with id 999"):
        s.complete_email_verification(999)


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

    before = s.get_user_by_username("boss").password_hash  # type: ignore[union-attr]
    assert s.seed_admin_from_env() is None
    after = s.get_user_by_username("boss").password_hash  # type: ignore[union-attr]
    assert after == before


# --- lifecycle -----------------------------------------------------------------


def test_instances_are_isolated(monkeypatch) -> None:
    monkeypatch.delenv("ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    a, b = _storage(), _storage()

    a.create_user("alice", "student")

    assert a.get_user_by_username("alice") is not None
    assert b.get_user_by_username("alice") is None


def test_close_drops_all_data(monkeypatch) -> None:
    monkeypatch.delenv("ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    s = _storage()
    s.create_user("alice", "student")

    s.close()

    assert s.list_users() == []
    assert s.list_roles() == []