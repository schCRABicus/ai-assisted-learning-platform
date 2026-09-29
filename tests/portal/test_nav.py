"""AppTest coverage for the portal's role-filtered navigation.

``st.navigation`` renders in the frontend, so an AppTest run has no sidebar links
to query. These tests instead capture the navigation mapping the entrypoint
hands to ``st.navigation`` and assert on that: which sections exist, which pages
are visible under each, and — separately — that every page is registered on
every run.

The registration half is not cosmetic. A page left out of the navigation stops
resolving, so its URL would 404 and any ``st.switch_page`` aimed at it would
raise for whoever is signed in at the time: ``views/student.py`` and
``views/verify.py`` both switch pages, and a signed-out visitor reaches the
second of them from an emailed link.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import streamlit
from streamlit.testing.v1 import AppTest
from unittest.mock import patch

from agentic_learning_portal import auth
from agentic_learning_portal.auth import get_storage

APP_PATH = Path(auth.__file__).parent / "app.py"

# Every page the portal serves, by title. The nav mapping is rebuilt on each run
# and the *visibility* differs per role, but the registered set never does —
# listing it here is what catches a page silently dropping out of a role's nav.
ALL_PAGES = {
    "Sign in",
    "Admin",
    "Assignments",
    "Users",
    "Student",
    "Assignment Editor",
    "Take assignment",
    "Verify account",
}


@contextmanager
def _capturing_navigation() -> Iterator[list[dict]]:
    """Record every ``st.navigation`` call made inside the block."""
    recorded: list[dict] = []
    real_navigation = streamlit.navigation

    def _capture(pages, **kwargs):
        recorded.append(pages)
        return real_navigation(pages, **kwargs)

    with patch("streamlit.navigation", _capture):
        yield recorded


def _app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=15)


def _sign_in(at: AppTest, username: str, password: str) -> AppTest:
    at.text_input[0].set_value(username)
    at.text_input[1].set_value(password)
    at.button[0].click().run()
    return at


def _visible(nav: dict) -> dict[str, list[str]]:
    """Section header -> the page titles the sidebar actually lists under it."""
    return {
        section: [p.title for p in pages if p.visibility == "visible"]
        for section, pages in nav.items()
    }


def _registered(nav: dict) -> set[str]:
    return {p.title for pages in nav.values() for p in pages}


def test_signed_out_sees_only_the_sign_in_entry(portal_env) -> None:
    with _capturing_navigation() as recorded:
        at = _app().run()

    assert not at.exception
    assert _visible(recorded[-1]) == {"Account": ["Sign in"]}
    # …and the page at ``/`` really is the sign-in form, not a guarded page
    # rendering one on its own behalf.
    assert any(t.label == "Username" for t in at.text_input)
    assert any(t.label == "Password" for t in at.text_input)


def _titles(at: AppTest) -> list[str]:
    return [t.value for t in at.title]


def test_admin_sees_only_the_admin_section(portal_env) -> None:
    with _capturing_navigation() as recorded:
        at = _app().run()
        _sign_in(at, "boss", "hunter2")

    assert not at.exception
    assert at.session_state["user"].username == "boss"
    assert _visible(recorded[-1]) == {"Admin": ["Admin", "Assignments", "Users"]}
    # Signing in on the public page forwards to the admin page.
    assert any("Task Generation Admin" in title for title in _titles(at))


def test_teacher_sees_only_the_admin_section(portal_env) -> None:
    get_storage().create_user("tess", "teacher", password="pw123")

    with _capturing_navigation() as recorded:
        at = _app().run()
        _sign_in(at, "tess", "pw123")

    assert not at.exception
    assert _visible(recorded[-1]) == {"Admin": ["Admin", "Assignments", "Users"]}


def test_student_sees_only_the_student_section(portal_env) -> None:
    get_storage().create_user("pupil", "student", password="pw123")

    with _capturing_navigation() as recorded:
        at = _app().run()
        _sign_in(at, "pupil", "pw123")

    assert not at.exception
    assert at.session_state["user"].username == "pupil"
    assert _visible(recorded[-1]) == {"Student": ["Student"]}
    # …and landing forwards to the student page, which has nothing to show yet.
    assert any("Student" in title for title in _titles(at))
    assert any("No assignments" in info.value for info in at.info)


def test_dual_role_user_sees_both_sections(portal_env) -> None:
    """Admin *and* student: both sections, admin first (the landing precedence)."""
    get_storage().create_user("both", ["admin", "student"], password="pw123")

    with _capturing_navigation() as recorded:
        at = _app().run()
        _sign_in(at, "both", "pw123")

    assert not at.exception
    assert _visible(recorded[-1]) == {
        "Admin": ["Admin", "Assignments", "Users"],
        "Student": ["Student"],
    }


def test_every_page_is_registered_on_every_run(portal_env) -> None:
    """The registered set is role-independent — only visibility changes.

    This is the invariant that keeps every URL and every ``st.switch_page``
    target resolvable no matter who is looking: for the run that resolves the
    page, ``st.navigation`` falls back to the default page hash when the
    intended page is absent, but a *named* URL (``/admin``) is matched against
    the registered pages and 404s when nothing matches.
    """
    get_storage().create_user("pupil", "student", password="pw123")
    get_storage().create_user("both", ["admin", "student"], password="pw123")

    with _capturing_navigation() as recorded:
        _app().run()  # signed out
        for username in ("boss", "pupil", "both"):
            at = _app().run()
            _sign_in(at, username, "pw123" if username != "boss" else "hunter2")

    assert recorded, "the entrypoint should have built a navigation"
    for nav in recorded:
        assert _registered(nav) == ALL_PAGES
