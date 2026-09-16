"""Assignment-taking endpoint of the Agentic Learning Portal.

Served at ``/take_assignment`` (a hidden page via ``st.navigation`` in
``app.py``), reached from the student page, which passes the assignment id
through session state (``take_assignment_id``).

The page is a small state machine over the student's attempts, driven by the
rules in ``assignment/attempts.py``:

- **no attempt yet, entitled to one** — a ▶️ Start button that opens an attempt;
- **an attempt in progress** — the answer form, one text input per task, with
  💾 Save progress (upserts the answers and stays ``in_progress``, so the student
  can leave and come back) and 📤 Submit for grading (saves, auto-grades through
  ``domains/math/grading.grade_attempt``, then completes the attempt);
- **attempts used up** — the graded results, read-only.

The lock is derived from storage on every run, never from session state, so a
submitted assignment cannot be reopened — only an admin granting another attempt
(``storage.grant_extra_attempt``) lets the student start a fresh, blank one.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from agentic_learning_portal.admin.formatting import latex_to_plain_text
from agentic_learning_portal.assignment import (
    active_attempt,
    attempt_score,
    can_start_attempt,
    student_attempts,
)
from agentic_learning_portal.auth import get_storage, require_roles
from agentic_learning_portal.domains.math import grade_attempt
from agentic_learning_portal.storage import Assignment, Attempt, Storage, Task

STUDENT_PAGE = str(Path(__file__).parent / "student.py")

# Set when an attempt is submitted, so the results view can greet the student
# with a success banner on the rerun that follows (and only that one).
_JUST_SUBMITTED = "take_just_submitted"


def _answer_key(attempt_id: int, task_id: int) -> str:
    """Return the session-state key holding the student's answer for a task."""
    return f"take_answer_{attempt_id}_{task_id}"


def _forget_answers(attempt_id: int) -> None:
    """Drop cached answer text for ``attempt_id`` (used when abandoning it)."""
    prefix = f"take_answer_{attempt_id}_"
    for key in [k for k in st.session_state if k.startswith(prefix)]:
        st.session_state.pop(key, None)


def _back_button() -> None:
    if st.button("← Back to my assignments", key="take_back"):
        st.switch_page(STUDENT_PAGE)


def _start_attempt(assignment_id: int, student_id: int) -> None:
    """Open a fresh attempt and reload into the answer form.

    ``get_storage()`` is called here rather than captured by the caller: this
    runs from a widget callback, and the sqlite backend hands out one instance
    per thread (a captured one would belong to a different thread).
    """
    get_storage().start_attempt(assignment_id, student_id)
    st.rerun()


def _save_answers(storage: Storage, attempt: Attempt, tasks: list[Task]) -> None:
    """Upsert the student's current answer for every task of the attempt.

    A blank input is stored as ``None`` rather than ``""``, so an untouched task
    is indistinguishable from a never-saved one.
    """
    for task in tasks:
        raw = st.session_state.get(_answer_key(attempt.id, task.id), "")
        storage.record_result(attempt.id, task.id, given_answer=raw.strip() or None)


def _render_breakdown(storage: Storage, attempt: Attempt, tasks: list[Task]) -> None:
    """Render the per-task given-vs-expected breakdown of a graded attempt."""
    answers = {
        result.task_id: result
        for result in storage.list_results(attempt_id=attempt.id)
    }
    correct, graded, average = attempt_score(storage, attempt)
    st.markdown(
        f"**📊 Score:** {correct}/{graded}"
        + (f" (avg {average:.2f})" if average is not None else "")
    )

    for index, task in enumerate(tasks):
        result = answers.get(task.id)
        given = result.given_answer if result is not None else None
        verdict = "✅" if result is not None and result.is_correct else "❌"
        st.markdown(
            f"{verdict} **Task {index + 1}** — {task.topic} · {task.complexity}"
        )
        st.markdown(latex_to_plain_text(task.text))
        st.markdown(f"- **Your answer:** {given if given else '_(not answered)_'}")
        st.markdown(f"- **Correct answer:** `{task.correct_answer}`")
        st.markdown("---")


def _render_start(storage: Storage, assignment: Assignment, student_id: int,
                  attempts: list[Attempt], tasks: list[Task]) -> None:
    """Render the pre-attempt screen offering to open the first (or a new) one."""
    if attempts:
        st.info(
            "Your previous attempt was submitted and graded. Your teacher has "
            "granted you another attempt — all answers start blank."
        )
    st.markdown(f"This assignment has **{len(tasks)} tasks**.")
    st.caption("You can save your progress as you go and submit when you're done.")
    if st.button("▶️ Start assignment", type="primary", key="take_start"):
        _start_attempt(assignment.id, student_id)


def _render_taking(storage: Storage, attempt: Attempt, tasks: list[Task]) -> None:
    """Render the answer form for an attempt in progress."""
    saved = {
        result.task_id: result.given_answer
        for result in storage.list_results(attempt_id=attempt.id)
    }

    for index, task in enumerate(tasks):
        st.markdown(f"**Task {index + 1} of {len(tasks)}** — {task.topic}")
        st.markdown(latex_to_plain_text(task.text))
        key = _answer_key(attempt.id, task.id)
        # Seed from storage only when the widget has no value yet: passing
        # ``value=`` alongside an existing session-state key is what Streamlit
        # warns about, and the widget's own value must win once the student types.
        if key not in st.session_state:
            st.session_state[key] = saved.get(task.id) or ""
        st.text_input(
            f"Your answer for task {index + 1}",
            key=key,
            label_visibility="collapsed",
            placeholder="Type your answer…",
        )
        st.markdown("---")

    col_save, col_submit = st.columns(2)
    if col_save.button("💾 Save progress", key="take_save", use_container_width=True):
        _save_answers(storage, attempt, tasks)
        st.success("Progress saved. You can close this page and come back later.")
    if col_submit.button(
        "📤 Submit for grading",
        key="take_submit",
        type="primary",
        use_container_width=True,
        help="Grades every task and locks the assignment — no more changes.",
    ):
        _save_answers(storage, attempt, tasks)
        grade_attempt(storage, attempt.id)
        storage.complete_attempt(attempt.id)
        _forget_answers(attempt.id)
        st.session_state[_JUST_SUBMITTED] = attempt.id
        st.rerun()


def _render_results(
    storage: Storage,
    assignment: Assignment,
    student_id: int,
    attempts: list[Attempt],
    tasks: list[Task],
) -> None:
    """Render the graded results, newest attempt first."""
    if st.session_state.pop(_JUST_SUBMITTED, None) is not None:
        st.success("Submitted! Here are your graded results.")
        st.balloons()

    total = len(attempts)
    for offset, attempt in enumerate(reversed(attempts)):
        label = "Latest attempt" if offset == 0 else f"Attempt {total - offset}"
        if total == 1:
            st.markdown(f"### {label}")
            _render_breakdown(storage, attempt, tasks)
        else:
            with st.expander(label, expanded=offset == 0):
                _render_breakdown(storage, attempt, tasks)

    if can_start_attempt(assignment, attempts):
        st.warning(
            "Your teacher has granted you another attempt. It starts blank, and "
            "the results above stay on record."
        )
        if st.button("🔁 Start new attempt", type="primary", key="take_start_new"):
            _start_attempt(assignment.id, student_id)
    else:
        st.info(
            "You have used every attempt for this assignment, so it can't be "
            "taken again. Ask your teacher if you need another one."
        )


@require_roles()
def render_student_take_page() -> None:
    st.set_page_config(page_title="📝 Take assignment", page_icon="📝")

    user = st.session_state["user"]
    assignment_id = st.session_state.get("take_assignment_id")
    if assignment_id is None:
        st.info("No assignment selected.")
        _back_button()
        st.stop()

    storage = get_storage()
    assignment = storage.get_assignment(assignment_id)
    # Session state is client-influenced, so never trust the id on its own: only
    # the student the assignment is actually assigned to may take it.
    if assignment is None or assignment.assigned_to != user.id:
        st.error("This assignment isn't available to you.")
        _back_button()
        st.stop()

    st.title(f"📝 {assignment.title}")

    tasks = storage.list_assignment_tasks(assignment.id)
    if not tasks:
        st.warning("This assignment has no tasks yet — check back once your teacher adds some.")
        _back_button()
        st.stop()

    attempts = student_attempts(storage, assignment.id, user.id)
    active = active_attempt(attempts)

    if active is not None:
        st.caption("Answer as many tasks as you like, then submit for grading.")
        _render_taking(storage, active, tasks)
    elif attempts:
        _render_results(storage, assignment, user.id, attempts, tasks)
    else:
        _render_start(storage, assignment, user.id, attempts, tasks)

    _back_button()


render_student_take_page()
