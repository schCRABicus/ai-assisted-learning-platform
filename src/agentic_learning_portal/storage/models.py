from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentic_learning_portal.api.model import GeneratedTask

RoleName = Literal["admin", "teacher", "student"]
AttemptStatus = Literal["in_progress", "completed"]


class Role(BaseModel):
    """A role that gates access within the portal."""

    id: int = Field(..., description="Role id.")
    name: RoleName = Field(..., description="Role name.")


class User(BaseModel):
    """A portal user who may hold several roles at once."""

    id: int = Field(..., description="User id.")
    username: str = Field(..., description="Unique login name.")
    roles: list[RoleName] = Field(
        ...,
        description="Roles the user holds; may be several (e.g. admin + teacher).",
    )
    email: str | None = Field(default=None, description="Optional email address.")
    password_hash: str | None = Field(
        default=None,
        repr=False,
        description="Salted scrypt hash of the login password, if set.",
    )
    created_at: str = Field(..., description="UTC ISO-8601 creation timestamp.")


class Task(GeneratedTask):
    """A generated task persisted in storage.

    Reuses the canonical ``GeneratedTask`` (topic/text/complexity/correct_answer/
    solution) and adds only the storage ``id``.
    """

    id: int = Field(..., description="Storage id of the task.")


class Assignment(BaseModel):
    """A collection of tasks assigned to a student."""

    id: int = Field(..., description="Assignment id.")
    title: str = Field(..., description="Human-readable title.")
    created_by: int = Field(..., description="Id of the teacher/admin who created it.")
    assigned_to: int | None = Field(
        default=None,
        description="Id of the student the assignment targets, if any.",
    )
    created_at: str = Field(..., description="UTC ISO-8601 creation timestamp.")


class Attempt(BaseModel):
    """A student's attempt at an assignment."""

    id: int = Field(..., description="Attempt id.")
    assignment_id: int = Field(..., description="Id of the assignment attempted.")
    student_id: int = Field(..., description="Id of the student attempting it.")
    status: AttemptStatus = Field(..., description="in_progress or completed.")
    started_at: str = Field(..., description="UTC ISO-8601 start timestamp.")
    completed_at: str | None = Field(
        default=None,
        description="UTC ISO-8601 completion timestamp, when completed.",
    )


class AttemptResult(BaseModel):
    """A scored answer for one task within an attempt."""

    id: int = Field(..., description="Result id.")
    attempt_id: int = Field(..., description="Id of the attempt.")
    task_id: int = Field(..., description="Id of the task answered.")
    given_answer: str | None = Field(
        default=None,
        description="The student's answer, as raw text.",
    )
    expected_answer: str | None = Field(
        default=None,
        description="The correct answer, as raw text.",
    )
    is_correct: bool | None = Field(
        default=None,
        description="Whether the answer was correct, when graded.",
    )
    score: float | None = Field(
        default=None,
        description="Partial credit awarded, when graded.",
    )
    detail: str = Field(default="", description="Grading detail or feedback.")