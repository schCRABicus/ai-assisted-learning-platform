from __future__ import annotations

from typing import Callable, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    email_verified: bool = Field(
        default=True,
        description="Whether the email address was verified via the invite link.",
    )
    password_hash: str | None = Field(
        default=None,
        repr=False,
        description="Salted scrypt hash of the login password, if set.",
    )
    verification_token_hash: str | None = Field(
        default=None,
        repr=False,
        description="Hash of the pending invite/verification token, if any.",
    )
    verification_expires_at: str | None = Field(
        default=None,
        description="UTC ISO-8601 expiry of the pending verification token, if any.",
    )
    created_at: str = Field(..., description="UTC ISO-8601 creation timestamp.")


class Task(GeneratedTask):
    """A generated task persisted in storage.

    Reuses the canonical ``GeneratedTask`` (topic/text/complexity/correct_answer/
    solution) and adds only the storage ``id``.
    """

    id: int = Field(..., description="Storage id of the task.")


class LazyTaskList:
    """A list-like adapter over an assignment's tasks.

    The number of tasks (``size``) is known up front — storage joins the task
    count when loading an assignment — but the task contents themselves are
    fetched lazily on first content access (iteration or indexing) and cached
    thereafter. Reading ``len()`` / ``size`` never triggers a fetch.
    """

    def __init__(self, size: int, loader: Callable[[], list[Task]]) -> None:
        self._size = size
        self._loader = loader
        self._items: list[Task] | None = None

    @property
    def size(self) -> int:
        """The known task count; never triggers a fetch."""
        return self._size

    @property
    def loaded(self) -> bool:
        """Whether the task contents have been fetched yet."""
        return self._items is not None

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Task]:
        return iter(self._load())

    def __getitem__(self, index: int | slice) -> Task | list[Task]:
        return self._load()[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LazyTaskList):
            return NotImplemented
        if self._size != other._size:
            return False
        if self._size == 0:
            return True
        return self._load() == other._load()

    def _load(self) -> list[Task]:
        if self._items is None:
            self._items = self._loader()
        return self._items


class Assignment(BaseModel):
    """A collection of tasks assigned to a student.

    ``tasks`` is a :class:`LazyTaskList`: its ``size`` is the assignment's task
    count (joined in when loading), while the task contents are fetched on
    first content access. A freshly created assignment has size 0.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int = Field(..., description="Assignment id.")
    title: str = Field(..., description="Human-readable title.")
    created_by: int = Field(..., description="Id of the teacher/admin who created it.")
    assigned_to: int | None = Field(
        default=None,
        description="Id of the student the assignment targets, if any.",
    )
    extra_attempts: int = Field(
        default=0,
        description=(
            "Additional attempts granted by the admin on top of the first one; "
            "a student may hold at most ``1 + extra_attempts`` attempts."
        ),
    )
    created_at: str = Field(..., description="UTC ISO-8601 creation timestamp.")
    tasks: LazyTaskList = Field(
        default_factory=lambda: LazyTaskList(0, lambda: []),
        description="Lazily-loaded task list; size known up front, contents on first access.",
    )


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
    results_seen: bool = Field(
        default=False,
        description="Whether the admin has viewed this attempt's graded results.",
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