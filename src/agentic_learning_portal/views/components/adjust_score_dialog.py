"""The adjust-score dialog (views/components/adjust_score_dialog.py).

:func:`adjust_score_dialog` is a Streamlit ``st.dialog`` opened by the assignment
editor page (``views/admin/02_assignment_editor.py``) through the
``adjust_score_open`` session-state flag, which holds the ``(attempt_id,
task_id)`` pair of the row being reviewed: the page sets the flag and re-calls
the dialog on every run while it is set, so the dialog survives the
``st.rerun()`` calls it makes.

It reads the row fresh from storage on every open and shows the task, the
student's answer and working, and the grade currently on record; **Save** writes
the override through ``storage.adjust_result``, **Cancel** leaves the row
untouched. Only the grading fields are written, so an override never disturbs the
answer or the solution the student gave.

The verdict and the score are separate on purpose: the verdict drives the ✅/❌
tally while the score is what the attempt's average is taken over, so a task
marked correct can still be worth partial credit.
"""

from __future__ import annotations

import streamlit as st

from agentic_learning_portal.admin.formatting import latex_to_plain_text, literal_lines
from agentic_learning_portal.storage import Storage
from agentic_learning_portal.storage.factory import StorageFactory

STORAGE_FACTORY = StorageFactory()

CORRECT = "✅ Correct"
INCORRECT = "❌ Incorrect"
_VERDICTS = (CORRECT, INCORRECT)


def get_storage() -> Storage:
    """Return the :class:`Storage` the portal should use.

    The backend is chosen by ``STORAGE_BACKEND`` (``"sqlite"`` or ``"memory"``),
    read at call time so tests and ``.env`` can switch it after import. The
    sqlite backend keeps one instance per thread (``seed_admin_from_env`` runs
    on construction, so the ``.env``-configured admin is available from the
    start); the memory backend is a single shared instance.
    """
    return STORAGE_FACTORY.get_storage()


def _verdict_key(attempt_id: int, task_id: int) -> str:
    """Session-state key for the dialog's verdict control on one row."""
    return f"adjust_score_verdict_{attempt_id}_{task_id}"


def _score_key(attempt_id: int, task_id: int) -> str:
    """Session-state key for the dialog's score control on one row."""
    return f"adjust_score_value_{attempt_id}_{task_id}"


def _close_adjust_score_dialog(attempt_id: int, task_id: int) -> None:
    """Close the dialog and rerun.

    The controls' session-state keys are dropped along with the open flag, so
    reopening the dialog for this row re-seeds them from storage rather than
    showing an edit that was cancelled or already saved.
    """
    st.session_state.pop("adjust_score_open", None)
    st.session_state.pop(_verdict_key(attempt_id, task_id), None)
    st.session_state.pop(_score_key(attempt_id, task_id), None)
    st.rerun()


@st.dialog("Adjust score", dismissible=True)
def adjust_score_dialog(attempt_id: int, task_id: int) -> None:
    """Override the verdict and score of one graded task.

    Reads the row fresh from storage on every open (so it reflects any earlier
    adjustment), and writes back through ``storage.adjust_result`` — grading
    fields only, never the student's answer or solution. Opened by the assignment
    editor page through the ``adjust_score_open`` session-state flag.
    """
    storage = get_storage()
    result = next(
        (
            row
            for row in storage.list_results(attempt_id=attempt_id)
            if row.task_id == task_id
        ),
        None,
    )
    if result is None:
        st.error(f"No result for task #{task_id} in attempt #{attempt_id}.")
        if st.button("✖ Close", key="adjust_score_close"):
            _close_adjust_score_dialog(attempt_id, task_id)
        return

    # The task may have been removed from the assignment since it was answered;
    # the result row outlives it, so render without the problem text then.
    task = storage.get_task(task_id)
    if task is not None:
        st.caption(f"{task.topic} · {task.complexity}")
        st.markdown(latex_to_plain_text(task.text))
        st.markdown("---")

    st.markdown(
        f"- **Student's answer:** {result.given_answer or '_(not answered)_'}"
    )
    st.markdown(
        "- **Student's solution:** "
        + (
            literal_lines(result.given_solution)
            if result.given_solution
            else "_(not given)_"
        )
    )
    st.markdown(f"- **Correct answer:** `{result.expected_answer or '—'}`")
    st.markdown("---")

    verdict_key = _verdict_key(attempt_id, task_id)
    if verdict_key not in st.session_state:
        st.session_state[verdict_key] = CORRECT if result.is_correct else INCORRECT
    verdict = st.radio(
        "Verdict",
        options=_VERDICTS,
        key=verdict_key,
        horizontal=True,
    )

    score_key = _score_key(attempt_id, task_id)
    if score_key not in st.session_state:
        st.session_state[score_key] = (
            result.score
            if result.score is not None
            else (1.0 if result.is_correct else 0.0)
        )
    score = st.number_input(
        "Score",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key=score_key,
        help="1.0 fully correct, 0.0 incorrect, or anything between for partial credit.",
    )
    st.caption(
        "The verdict sets the ✅/❌ tally; the score is what the attempt's "
        "average is taken over."
    )

    col_save, col_cancel = st.columns(2)
    if col_save.button("💾 Save", type="primary", key="adjust_score_save"):
        label = task.topic if task is not None else f"task #{task_id}"
        storage.adjust_result(
            attempt_id,
            task_id,
            is_correct=verdict == CORRECT,
            score=float(score),
        )
        # Hand the confirmation to the page: the dialog closes on the rerun
        # below, so a message rendered here would never be seen.
        st.session_state["editor_adjusted_score"] = (label, verdict == CORRECT, score)
        _close_adjust_score_dialog(attempt_id, task_id)
    if col_cancel.button("✖ Cancel", key="adjust_score_cancel"):
        _close_adjust_score_dialog(attempt_id, task_id)
