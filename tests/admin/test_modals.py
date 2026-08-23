"""Tests for the shared confirmation dialogs (views/components/modals.py).

The dialog is exercised in isolation from the assignments page through a tiny
AppTest harness that opens the real ``delete_assignment_dialog`` for a seeded
assignment. The harness calls the dialog unconditionally because AppTest only
renders a ``st.dialog`` body on runs where its opening call is active — with an
``if``-guarded call the confirm click would never be processed.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from agentic_learning_portal.auth import get_storage

# ``__AID__`` is replaced with the seeded assignment id per test.
HARNESS = """
import streamlit as st

from agentic_learning_portal.views.components.modals import delete_assignment_dialog

delete_assignment_dialog(__AID__)
"""


def _boss_id() -> int:
    """Return the env-seeded admin's user id."""
    return get_storage().get_user_by_username("boss").id


def _open_dialog(assignment_id: int) -> AppTest:
    """Run the harness and return the AppTest with the dialog open."""
    at = AppTest.from_string(
        HARNESS.replace("__AID__", str(assignment_id)), default_timeout=10
    )
    at.run()
    assert not at.exception
    return at


def test_dialog_renders_confirmation_prompt(portal_env) -> None:
    """The dialog explains the irreversible delete before offering buttons."""
    assignment = get_storage().create_assignment("doomed", created_by=_boss_id())

    at = _open_dialog(assignment.id)

    assert any(
        "Are you sure you want to delete this assignment?" in m.value
        for m in at.markdown
    )
    labels = {b.label for b in at.button}
    assert "Yes, Delete" in labels
    assert "Cancel" in labels


def test_dialog_yes_deletes_the_assignment(portal_env) -> None:
    """Confirming removes the assignment from storage."""
    assignment = get_storage().create_assignment("doomed", created_by=_boss_id())
    at = _open_dialog(assignment.id)

    yes = [b for b in at.button if b.label == "Yes, Delete"]
    assert yes, "expected the Yes, Delete button"
    yes[0].click().run()

    assert not at.exception
    assert get_storage().get_assignment(assignment.id) is None


def test_dialog_cancel_preserves_the_assignment(portal_env) -> None:
    """Dismissing with Cancel leaves the assignment in storage."""
    assignment = get_storage().create_assignment("kept", created_by=_boss_id())
    at = _open_dialog(assignment.id)

    cancel = [b for b in at.button if b.label == "Cancel"]
    assert cancel, "expected the Cancel button"
    cancel[0].click().run()

    assert not at.exception
    assert get_storage().get_assignment(assignment.id) is not None
