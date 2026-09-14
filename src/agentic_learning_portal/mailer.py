"""Invite-email delivery for the portal.

The mailer is a deliberately swappable seam. ``send_invite_email`` is a dev
stub: it logs the verification URL instead of talking to a real provider, so the
admin "invite a user" flow works end to end with no credentials. A real provider
(SendGrid, SES, …) can be dropped in behind the same signature — the rest of the
portal depends only on ``build_verify_url`` / ``send_invite_email``, never on
how the email actually leaves the process.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# The public base URL the verification link is built against. Override with
# ``PORTAL_BASE_URL`` (e.g. a deployed origin); the default matches ``run-portal``.
DEFAULT_BASE_URL = "http://localhost:8501"


def build_verify_url(token: str) -> str:
    """Return the public verification URL for a raw ``token``.

    The token is the *only* copy of the raw secret (storage holds its SHA-256
    hash); it appears here, embedded in the URL, and nowhere else.
    """
    base = os.getenv("PORTAL_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    return f"{base}/verify?token={token}"


def send_invite_email(email: str, verify_url: str) -> str:
    """Deliver an invite to ``email`` and return the verification URL.

    Dev stub: logs the URL (and ``email``) rather than sending a real message.
    Returns the URL so the admin page can surface it in the UI while email is
    stubbed.
    """
    logger.info("Invite for %s — verification link: %s", email, verify_url)
    return verify_url