"""AppTest-driven tests for the public account-verification page.

The page (``views/verify.py``) is the one unauthenticated endpoint: it reads a
one-time ``?token=`` query param, lets the invited user set a password, and
marks their email verified. Tests run against the in-memory backend
(``portal_env``), create an unverified user + token on the main thread, and drive
the page through ``AppTest`` with ``query_params``.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from agentic_learning_portal.auth import get_storage

VERIFY_PAGE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "agentic_learning_portal"
    / "views"
    / "verify.py"
)


def _invite() -> tuple[int, str]:
    """Create an unverified user and return ``(user_id, raw_token)``."""
    storage = get_storage()
    user = storage.create_user(
        "invitee", "student", email="invitee@example.com", email_verified=False
    )
    return user.id, storage.issue_verification_token(user.id)


def _verify_page(token: str | None) -> AppTest:
    at = AppTest.from_file(str(VERIFY_PAGE), default_timeout=10)
    if token is not None:
        at.query_params = {"token": token}
    at.run()
    return at


def test_missing_token_shows_error(portal_env) -> None:
    at = _verify_page(None)

    assert not at.exception
    assert any("missing its token" in e.value for e in at.error)


def test_unknown_token_shows_error(portal_env) -> None:
    at = _verify_page("does-not-exist")

    assert not at.exception
    assert any("invalid or has expired" in e.value for e in at.error)


def test_valid_token_renders_password_form(portal_env) -> None:
    _user_id, token = _invite()

    at = _verify_page(token)

    assert not at.exception
    labels = {t.label for t in at.text_input}
    assert "Password" in labels
    assert "Confirm password" in labels
    assert any("Set password" in b.label for b in at.button)


def test_submit_sets_password_and_verifies(portal_env) -> None:
    storage = get_storage()
    user_id, token = _invite()

    at = _verify_page(token)

    password = [t for t in at.text_input if t.label == "Password"][0]
    confirm = [t for t in at.text_input if t.label == "Confirm password"][0]
    password.set_value("newpass123")
    confirm.set_value("newpass123")

    submit = [b for b in at.button if "Set password" in b.label][0]
    submit.click().run()

    assert not at.exception
    refreshed = storage.get_user(user_id)
    assert refreshed.email_verified is True
    assert refreshed.verification_token_hash is None
    assert refreshed.verification_expires_at is None
    assert storage.verify_credentials("invitee", "newpass123") is not None
    assert any("Account verified" in s.value for s in at.success)


def test_mismatched_passwords_are_rejected(portal_env) -> None:
    storage = get_storage()
    user_id, token = _invite()

    at = _verify_page(token)

    password = [t for t in at.text_input if t.label == "Password"][0]
    confirm = [t for t in at.text_input if t.label == "Confirm password"][0]
    password.set_value("newpass123")
    confirm.set_value("different123")

    submit = [b for b in at.button if "Set password" in b.label][0]
    submit.click().run()

    assert not at.exception
    assert any("do not match" in e.value for e in at.error)
    # Nothing was persisted: still unverified, no password.
    refreshed = storage.get_user(user_id)
    assert refreshed.email_verified is False
    assert refreshed.password_hash is None