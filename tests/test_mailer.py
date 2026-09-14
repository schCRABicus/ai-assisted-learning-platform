"""Tests for the invite mailer (agentic_learning_portal/mailer.py).

The mailer is a dev stub — it never touches the network, so these tests need no
seams. They pin the URL shape and the "returns the link" contract the rest of
the portal relies on.
"""

from __future__ import annotations

from agentic_learning_portal.mailer import build_verify_url, send_invite_email


def test_build_verify_url_uses_default_base() -> None:
    assert build_verify_url("abc123") == "http://localhost:8501/verify?token=abc123"


def test_build_verify_url_uses_portal_base_url(monkeypatch) -> None:
    monkeypatch.setenv("PORTAL_BASE_URL", "https://portal.example.com")

    assert build_verify_url("tok") == "https://portal.example.com/verify?token=tok"


def test_build_verify_url_strips_trailing_slash(monkeypatch) -> None:
    monkeypatch.setenv("PORTAL_BASE_URL", "https://portal.example.com/")

    assert build_verify_url("tok") == "https://portal.example.com/verify?token=tok"


def test_send_invite_email_returns_the_link() -> None:
    url = "http://localhost:8501/verify?token=tok"

    assert send_invite_email("invitee@example.com", url) == url