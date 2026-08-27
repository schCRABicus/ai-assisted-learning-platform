"""Tests for the task-authoring dialog (views/components/create_task_dialog.py).

The dialog is exercised in isolation from the pages through a tiny AppTest
harness that opens the real dialog for a seeded assignment. The harness calls
the dialog unconditionally because AppTest only renders a ``st.dialog`` body on
runs where its opening call is active — with an ``if``-guarded call the confirm
click would never be processed.

The dialog is driven through :data:`CREATE_HARNESS`: tests type into the
authoring form, patch ``MathProblemGenerator.generate`` (so no real LLM call
escapes the ``_forbid_real_llm_calls`` guard), click Generate, and assert the
phases the dialog walks through. Generation runs in a real worker thread, so
these tests also exercise the live-progress path: the dialog's poll loop re-runs
until the worker sets ``result``/``error``, and the collected ``log`` is what
the progress UI renders.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from streamlit.testing.v1 import AppTest

from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.api.progress import ProgressEvent
from agentic_learning_portal.auth import get_storage
from agentic_learning_portal.domains.math import MathProblemGenerator

# ``__AID__`` is replaced with the seeded assignment id per test.
CREATE_HARNESS = """
import streamlit as st

from agentic_learning_portal.views.components.create_task_dialog import create_task_dialog

create_task_dialog(__AID__)
"""

_TASK = GeneratedTask(
    topic="Algebra",
    text="What is 2 + 2?",
    complexity="easy",
    correct_answer="4",
    solution="Add the numbers.",
)


def _boss_id() -> int:
    """Return the env-seeded admin's user id."""
    return get_storage().get_user_by_username("boss").id


def _open_create_dialog(assignment_id: int) -> AppTest:
    """Run the create-task harness and return the AppTest with the dialog open."""
    at = AppTest.from_string(
        CREATE_HARNESS.replace("__AID__", str(assignment_id)), default_timeout=20
    )
    at.run()
    assert not at.exception
    return at


def _find_button(at: AppTest, label: str) -> AppTest | None:
    """Return the first button whose label contains ``label`` (case-insensitive)."""
    needle = label.lower()
    for button in at.button:
        if needle in button.label.lower():
            return button
    return None


def _find_text_input(at: AppTest, label: str) -> AppTest | None:
    """Return the first text input whose label contains ``label``."""
    needle = label.lower()
    for widget in at.text_input:
        if needle in widget.label.lower():
            return widget
    return None


def _fill_form(at: AppTest, topic: str = "Algebra", manual: str = "linear equations") -> None:
    """Type a topic + manual subtopics so the Generate button becomes enabled."""
    topic_input = _find_text_input(at, "topic")
    assert topic_input is not None
    topic_input.set_value(topic).run()
    manual_input = _find_text_input(at, "subtopics")
    assert manual_input is not None
    manual_input.set_value(manual).run()


# --- create-task dialog ----------------------------------------------------------


def test_create_dialog_renders_authoring_form(portal_env) -> None:
    """The dialog opens with the authoring form and Generate disabled."""
    assignment = get_storage().create_assignment("author", created_by=_boss_id())

    at = _open_create_dialog(assignment.id)

    assert not at.exception
    assert _find_text_input(at, "topic") is not None
    assert _find_text_input(at, "context") is not None
    assert any("subtopics" in ms.label.lower() for ms in at.multiselect)
    generate = _find_button(at, "🎯 Generate")
    assert generate is not None and generate.disabled
    assert _find_button(at, "✖ Cancel") is not None


def test_generate_enables_after_topic_and_subtopic(portal_env) -> None:
    """Generate stays disabled until a topic and at least one subtopic exist."""
    assignment = get_storage().create_assignment("enable", created_by=_boss_id())
    at = _open_create_dialog(assignment.id)

    generate = _find_button(at, "🎯 Generate")
    assert generate is not None and generate.disabled

    # Topic alone is not enough — the suggestion LLM is blocked in tests, so the
    # dialog warns and keeps the manual-entry path.
    _find_text_input(at, "topic").set_value("Algebra").run()
    assert any("could not suggest subtopics" in w.value.lower() for w in at.warning)

    generate = _find_button(at, "🎯 Generate")
    assert generate is not None and generate.disabled

    # Adding a manual subtopic enables generation.
    _find_text_input(at, "subtopics").set_value("linear equations").run()
    generate = _find_button(at, "🎯 Generate")
    assert generate is not None and not generate.disabled


def test_listener_events_flow_into_progress_ui(portal_env) -> None:
    """The generator's progress events reach the listener and are surfaced.

    This is the heart of the "listener works / progress is always shown"
    requirement: a patched generator emits ``generate``/``validate``/``done``
    events through the ``listener=`` kwarg, and the dialog's
    :class:`CollectingProgressListener` must append them to the session-state
    ``log`` that the progress UI renders.
    """
    assignment = get_storage().create_assignment("listen", created_by=_boss_id())
    at = _open_create_dialog(assignment.id)
    _fill_form(at)

    async def _generate_emitting(prompt_input, **kwargs):
        listener = kwargs.get("listener")
        if listener is not None:
            await listener.on_progress(
                ProgressEvent("generate", "Generating task...", 1, 1)
            )
            await listener.on_progress(
                ProgressEvent("validate", "Schema validated", 1, 1)
            )
            await listener.on_progress(
                ProgressEvent("done", "Task generated and verified", 1, 1)
            )
        return _TASK

    with patch.object(
        MathProblemGenerator, "generate", new=AsyncMock(side_effect=_generate_emitting)
    ):
        _find_button(at, "🎯 Generate").click().run()

    assert not at.exception
    state = at.session_state["create_task_state"]
    # Every event the generator emitted through its listener was collected.
    assert [e.stage for e in state["log"]] == ["generate", "validate", "done"]
    assert "Task generated and verified" in " ".join(e.message for e in state["log"])
    # The result phase surfaces the collected log and the finished task card.
    assert any("Generation progress" in e.label for e in at.expander)
    assert "2 + 2" in " ".join(m.value for m in at.markdown)


def test_generation_failure_shows_error_and_log(portal_env) -> None:
    """A RuntimeError from the generator lands in the error phase with the log."""
    assignment = get_storage().create_assignment("fail", created_by=_boss_id())
    at = _open_create_dialog(assignment.id)
    _fill_form(at)

    with patch.object(
        MathProblemGenerator,
        "generate",
        new=AsyncMock(side_effect=RuntimeError("Max retries reached")),
    ):
        _find_button(at, "🎯 Generate").click().run()

    assert not at.exception
    assert any("Task generation failed" in e.value for e in at.error)
    # The log is still surfaced alongside the error.
    assert any("Generation progress" in e.label for e in at.expander)
    assert _find_button(at, "↻ Try again") is not None
    assert _find_button(at, "✖ Cancel") is not None


def test_save_persists_task_and_links_to_assignment(portal_env) -> None:
    """The Save button persists the generated task and links it to the assignment."""
    assignment = get_storage().create_assignment("save", created_by=_boss_id())
    at = _open_create_dialog(assignment.id)
    _fill_form(at)

    with patch.object(
        MathProblemGenerator, "generate", new=AsyncMock(return_value=_TASK)
    ):
        _find_button(at, "🎯 Generate").click().run()

    save = [b for b in at.button if b.label == "💾 Save"]
    assert save, "expected the Save button"
    save[0].click().run()

    assert not at.exception
    tasks = get_storage().list_assignment_tasks(assignment.id)
    assert len(tasks) == 1
    assert tasks[0].topic == "Algebra"
    # The dialog state was reset: back to a fresh authoring form.
    state = at.session_state["create_task_state"]
    assert state["result"] is None
    assert state["error"] is None


def test_cancel_clears_dialog_state(portal_env) -> None:
    """Cancel closes the dialog and drops its state (nothing is persisted)."""
    assignment = get_storage().create_assignment("cancel", created_by=_boss_id())
    at = _open_create_dialog(assignment.id)

    cancel = [b for b in at.button if b.label == "✖ Cancel"]
    assert cancel, "expected the Cancel button"
    cancel[0].click().run()

    assert not at.exception
    assert "create_task_open" not in at.session_state
    # The dialog state was dropped and recreated fresh.
    state = at.session_state["create_task_state"]
    assert state["result"] is None
    assert state["error"] is None
    assert get_storage().list_assignment_tasks(assignment.id) == []
