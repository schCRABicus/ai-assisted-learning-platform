from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ADMIN_PAGE = (
    Path(__file__).resolve().parents[2] / "src" / "agentic_learning_portal" / "pages" / "admin.py"
)


def test_admin_page_boots_and_renders_form() -> None:
    at = AppTest.from_file(str(ADMIN_PAGE), default_timeout=10)
    at.run()

    assert not at.exception
    assert at.title[0].value == "🎓 Task Generation Admin"

    # Without a topic, generation is disabled and the result section is absent.
    generate_button = at.button[0]
    assert generate_button.disabled
    assert at.subheader == []