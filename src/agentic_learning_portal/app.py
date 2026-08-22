"""Single Streamlit entry point for the Agentic Learning Portal.

Admin and (future) student views are exposed as distinct endpoints via
``st.navigation`` (``/admin``, ``/student``). Launch with ``uv run run-portal``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agentic_learning_portal.auth import render_sidebar_user

# Load API keys (e.g. GOOGLE_API_KEY) from .env before any page runs.
load_dotenv()

PAGES_DIR = Path(__file__).parent / "views"

st.set_page_config(page_title="Agentic Learning Portal", page_icon="🎓", layout="wide")

admin = st.Page(
    str(PAGES_DIR / "admin.py"),
    title="Admin",
    icon="🎓",
    url_path="admin",
    default=True,
)
assignments = st.Page(
    str(PAGES_DIR / "admin/assignments.py"),
    title="Assignments",
    icon="🎓",
    url_path="assignments",
    default=False,
)
student = st.Page(
    str(PAGES_DIR / "student.py"),
    title="Student",
    icon="🧑‍🎓",
    url_path="student",
)

# Show the signed-in user + Log out in the sidebar on every page. Pages guard
# themselves (see ``auth.require_roles``), so this is cosmetic, not a gate.
render_sidebar_user()

pg = st.navigation({
    "Admin": [admin, assignments],
    "Student": [student]
})
pg.run()


def main() -> None:
    """Launch the portal with Streamlit's CLI."""
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    from streamlit.web import cli

    cli.main()