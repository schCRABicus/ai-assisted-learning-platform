from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ADMIN_PAGE = (
    Path(__file__).resolve().parents[2] / "src" / "agentic_learning_portal" / "pages" / "admin.py"
)


def test_admin_page_boots_and_renders_form(portal_env) -> None:
    at = AppTest.from_file(str(ADMIN_PAGE), default_timeout=10)
    at.run()

    assert not at.exception
    # The page is auth-gated, so an anonymous run shows the login form, not the
    # admin form. Sign in with the env-seeded admin (portal_env), then the page
    # re-renders as the admin form.
    at.text_input[0].set_value("boss")
    at.text_input[1].set_value("hunter2")
    at.button[0].click().run()

    assert not at.exception
    assert at.title[0].value == "🎓 Task Generation Admin"

    # Without a topic, generation is disabled and the result section is absent.
    generate_button = at.button[0]
    assert generate_button.disabled
    assert at.subheader == []