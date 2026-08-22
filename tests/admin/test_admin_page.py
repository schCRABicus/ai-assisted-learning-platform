from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from streamlit.testing.v1 import AppTest

from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.domains.math import MathProblemGenerator

ADMIN_PAGE = (
    Path(__file__).resolve().parents[2] / "src" / "agentic_learning_portal" / "pages" / "admin.py"
)


def _login(at: AppTest) -> None:
    """Sign in with the env-seeded admin (portal_env) through the login form."""
    at.text_input[0].set_value("boss")
    at.text_input[1].set_value("hunter2")
    at.button[0].click().run()


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


def test_admin_page_boots_and_renders_assignment_form(portal_env) -> None:
    """The landing view is just the assignment-creation form (title + Add)."""
    at = AppTest.from_file(str(ADMIN_PAGE), default_timeout=10)
    at.run()

    assert not at.exception
    # The page is auth-gated, so an anonymous run shows the login form, not the
    # assignment form. Sign in with the env-seeded admin (portal_env), then the
    # page re-renders as the assignment form.
    _login(at)

    assert not at.exception
    assert at.title[0].value == "🎓 Task Generation Admin"
    assert any("Create an assignment" in s.value for s in at.subheader)

    title_input = _find_text_input(at, "assignment title")
    add_button = _find_button(at, "Add assignment")
    assert title_input is not None, "Assignment title input should be present"
    assert add_button is not None, "Add assignment button should be present"
    assert not title_input.disabled, "Title input should be enabled"
    # No title yet -> the add button is disabled.
    assert add_button.disabled


class TestAssignmentCarousel:
    """Test the assignment carousel feature in the admin page."""

    def _start_assignment(self, at: AppTest, title: str = "Week 3 Algebra Practice") -> None:
        title_input = _find_text_input(at, "assignment title")
        assert title_input is not None
        title_input.set_value(title)
        add_button = _find_button(at, "Add assignment")
        assert add_button is not None
        add_button.click().run()

    def test_add_assignment_creates_and_shows_carousel(self, portal_env) -> None:
        """Adding an assignment persists it and swaps to the carousel."""
        at = AppTest.from_file(str(ADMIN_PAGE), default_timeout=10)
        at.run()
        _login(at)
        self._start_assignment(at, "Week 3 Algebra Practice")

        assert not at.exception
        # Assignment active in session state.
        assert at.session_state["assignment_active"] is True
        assert at.session_state["assignment_id"] is not None
        assert at.session_state["assignment_title"] == "Week 3 Algebra Practice"

        # Carousel shows the task counter for the first slot.
        assert any("Task 1 of 1" in m.value for m in at.markdown)

        # Assignment persisted in storage with the current user as creator.
        from agentic_learning_portal.auth import get_storage

        storage = get_storage()
        assignments = storage.list_assignments()
        assert len(assignments) == 1
        assert assignments[0].title == "Week 3 Algebra Practice"
        assert assignments[0].created_by == at.session_state["user"].id

        # The landing form is gone in carousel mode: no title input, no add button.
        assert _find_text_input(at, "assignment title") is None
        assert _find_button(at, "Add assignment") is None

        # Cancel + Finish are available instead.
        assert _find_button(at, "Cancel assignment") is not None
        assert _find_button(at, "Finish") is not None

    def test_carousel_add_task_increases_slot_count(self, portal_env) -> None:
        """'+ Add task' (the plus to the right) appends a slot and navigates to it."""
        at = AppTest.from_file(str(ADMIN_PAGE), default_timeout=10)
        at.run()
        _login(at)
        self._start_assignment(at, "Test Assignment")

        assert at.session_state["assignment_task_count"] == 1

        add_button = _find_button(at, "Add task")
        assert add_button is not None
        add_button.click().run()

        assert at.session_state["assignment_task_count"] == 2
        assert at.session_state["assignment_current_index"] == 1
        assert any("Task 2 of 2" in m.value for m in at.markdown)

    def test_carousel_navigation(self, portal_env) -> None:
        """◀ / ▶ arrows flanking the slide navigate between slots."""
        at = AppTest.from_file(str(ADMIN_PAGE), default_timeout=10)
        at.run()
        _login(at)
        self._start_assignment(at, "Navigation Test")

        # Add two more slots (3 total), ending on index 2.
        for _ in range(2):
            _find_button(at, "Add task").click().run()
        assert at.session_state["assignment_task_count"] == 3
        assert at.session_state["assignment_current_index"] == 2

        # On the last slot only ◀ (previous) is available.
        assert _find_button(at, "◀") is not None
        assert _find_button(at, "▶") is None

        # Navigate back to the middle slot.
        _find_button(at, "◀").click().run()
        assert at.session_state["assignment_current_index"] == 1
        assert _find_button(at, "◀") is not None
        assert _find_button(at, "▶") is not None

        # Back to the first slot -> no ◀.
        _find_button(at, "◀").click().run()
        assert at.session_state["assignment_current_index"] == 0
        assert _find_button(at, "◀") is None
        assert _find_button(at, "▶") is not None

    def test_empty_slot_shows_generate_button_but_no_form(self, portal_env) -> None:
        """Generation forms are hidden until 'Generate task' is clicked."""
        at = AppTest.from_file(str(ADMIN_PAGE), default_timeout=10)
        at.run()
        _login(at)
        self._start_assignment(at, "Hidden Form Test")

        assert not at.exception
        # The empty slot offers the Generate entry point, not the form itself.
        assert _find_button(at, "Generate task") is not None
        assert _find_text_input(at, "topic") is None
        assert _find_button(at, "🎯 Generate") is None

    def test_generate_button_opens_embedded_form(self, portal_env) -> None:
        """Clicking 'Generate task' reveals the authoring form in the slide."""
        at = AppTest.from_file(str(ADMIN_PAGE), default_timeout=10)
        at.run()
        _login(at)
        self._start_assignment(at, "Form Open Test")

        _find_button(at, "Generate task").click().run()

        assert not at.exception
        # The form is now embedded in the slide (topic field, submit, discard).
        assert _find_text_input(at, "topic") is not None
        assert _find_button(at, "🎯 Generate") is not None
        assert _find_button(at, "Discard") is not None

    def test_typing_topic_degrades_gracefully_when_suggestion_fails(self, portal_env) -> None:
        """A failed subtopic suggestion warns and keeps the manual-entry path."""
        at = AppTest.from_file(str(ADMIN_PAGE), default_timeout=10)
        at.run()
        _login(at)
        self._start_assignment(at, "Subtopic Test")

        _find_button(at, "Generate task").click().run()
        topic_input = _find_text_input(at, "topic")
        assert topic_input is not None
        topic_input.set_value("Algebra").run()

        assert not at.exception
        # The LLM suggestion is blocked in tests, so the page must degrade
        # gracefully instead of crashing.
        assert any("could not suggest subtopics" in w.value.lower() for w in at.warning)
        # The subtopics multiselect is still present (empty options) so the
        # admin can add subtopics manually.
        assert any("subtopics" in ms.label.lower() for ms in at.multiselect)

    @staticmethod
    def _task(topic: str, text: str) -> GeneratedTask:
        return GeneratedTask(
            topic=topic,
            text=text,
            complexity="easy",
            correct_answer="4",
            solution="Work it out step by step.",
        )

    def _open_form_and_fill(self, at: AppTest, topic: str, manual: str) -> None:
        """Open the embedded form for the current slot and set topic + subtopics."""
        _find_button(at, "Generate task").click().run()
        _find_text_input(at, "topic").set_value(topic).run()
        _find_text_input(at, "subtopics").set_value(manual).run()

    def test_generated_task_survives_navigation(self, portal_env) -> None:
        """A generated task still shows when navigating away and back."""
        at = AppTest.from_file(str(ADMIN_PAGE), default_timeout=10)
        at.run()
        _login(at)
        self._start_assignment(at, "Persist Test")

        self._open_form_and_fill(at, "Algebra", "linear equations")
        with patch.object(
            MathProblemGenerator, "generate", new=AsyncMock(return_value=self._task("Algebra", "What is 2 + 2?"))
        ):
            _find_button(at, "🎯 Generate").click().run()

        assert not at.exception
        assert at.session_state["carousel_last_task_0"] is not None
        # The form is gone; the card is shown.
        assert _find_text_input(at, "topic") is None
        assert "2 + 2" in " ".join(m.value for m in at.markdown)

        # Add a slot -> lands on slot 1 (empty).
        _find_button(at, "Add task").click().run()
        assert at.session_state["assignment_current_index"] == 1
        assert at.session_state["carousel_last_task_0"] is not None

        # Navigate back to slot 0: the card must still be there.
        _find_button(at, "◀").click().run()
        assert at.session_state["assignment_current_index"] == 0
        assert at.session_state["carousel_last_task_0"] is not None
        assert "2 + 2" in " ".join(m.value for m in at.markdown)

    def test_multiple_generated_tasks_survive_navigation(self, portal_env) -> None:
        """Each slot keeps its own task when navigating between generated slots."""
        at = AppTest.from_file(str(ADMIN_PAGE), default_timeout=10)
        at.run()
        _login(at)
        self._start_assignment(at, "Two Slot Test")

        # Slot 0: Algebra task.
        self._open_form_and_fill(at, "Algebra", "linear equations")
        with patch.object(
            MathProblemGenerator, "generate", new=AsyncMock(return_value=self._task("Algebra", "What is 2 + 2?"))
        ):
            _find_button(at, "🎯 Generate").click().run()

        # Add slot 1 and generate a Geometry task there.
        _find_button(at, "Add task").click().run()
        self._open_form_and_fill(at, "Geometry", "triangles")
        with patch.object(
            MathProblemGenerator, "generate", new=AsyncMock(return_value=self._task("Geometry", "Triangle area?"))
        ):
            _find_button(at, "🎯 Generate").click().run()
        assert at.session_state["assignment_current_index"] == 1
        assert at.session_state["carousel_last_task_1"] is not None

        # On slot 1 the Geometry card shows, not the Algebra one.
        assert "Triangle area?" in " ".join(m.value for m in at.markdown)
        assert "2 + 2" not in " ".join(m.value for m in at.markdown)

        # Back to slot 0: the Algebra card returns.
        _find_button(at, "◀").click().run()
        assert at.session_state["assignment_current_index"] == 0
        assert "2 + 2" in " ".join(m.value for m in at.markdown)
        assert "Triangle area?" not in " ".join(m.value for m in at.markdown)

        # And forward again to slot 1.
        _find_button(at, "▶").click().run()
        assert "Triangle area?" in " ".join(m.value for m in at.markdown)

    def test_storage_backfill_restores_task_into_slot(self, portal_env) -> None:
        """A slot whose session state was lost is backfilled from storage."""
        at = AppTest.from_file(str(ADMIN_PAGE), default_timeout=10)
        at.run()
        _login(at)
        self._start_assignment(at, "Backfill Test")

        self._open_form_and_fill(at, "Algebra", "linear equations")
        with patch.object(
            MathProblemGenerator, "generate", new=AsyncMock(return_value=self._task("Algebra", "What is 2 + 2?"))
        ):
            _find_button(at, "🎯 Generate").click().run()
        assert at.session_state["carousel_last_task_0"] is not None

        # Simulate session churn: the in-session slot task is lost, but the
        # assignment + persisted task remain in storage.
        at.session_state["carousel_last_task_0"] = None
        at.run()

        assert not at.exception
        # The storage backfill restored the task into the slot.
        assert at.session_state["carousel_last_task_0"] is not None
        assert "2 + 2" in " ".join(m.value for m in at.markdown)

    def test_cancel_assignment_clears_state(self, portal_env) -> None:
        """Cancel resets assignment state and returns to the landing form."""
        at = AppTest.from_file(str(ADMIN_PAGE), default_timeout=10)
        at.run()
        _login(at)
        self._start_assignment(at, "Cancel Test")
        assert at.session_state["assignment_active"] is True

        cancel_button = _find_button(at, "Cancel assignment")
        assert cancel_button is not None
        cancel_button.click().run()

        assert at.session_state["assignment_active"] is False
        assert at.session_state["assignment_id"] is None
        assert at.session_state["assignment_title"] == ""
        assert at.session_state["assignment_task_count"] == 0
        assert at.session_state["assignment_current_index"] == 0

        # Back to the landing form: title input enabled, add button present.
        fresh_title = _find_text_input(at, "assignment title")
        assert fresh_title is not None and not fresh_title.disabled
        assert _find_button(at, "Add assignment") is not None

    def test_carousel_finish_assignment_shows_summary(self, portal_env) -> None:
        """Finish clears state and returns to the landing form."""
        at = AppTest.from_file(str(ADMIN_PAGE), default_timeout=10)
        at.run()
        _login(at)
        self._start_assignment(at, "Finish Test")

        finish_button = _find_button(at, "Finish")
        assert finish_button is not None
        finish_button.click().run()

        assert not at.exception
        # Finishing clears the active carousel.
        assert at.session_state["assignment_active"] is False
        assert at.session_state["assignment_id"] is None

        # The landing form is reachable again.
        assert _find_button(at, "Add assignment") is not None