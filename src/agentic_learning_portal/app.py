"""Single Streamlit entry point for the Agentic Learning Portal.

Each view is exposed as its own endpoint via ``st.navigation``: the sign-in page
at ``/`` (the one ``default`` page), the admin pages under ``/admin``,
``/assignments`` and ``/users``, and the student pages under ``/student``.
Launch with ``uv run run-portal``.

The sidebar is built per run from the signed-in user's roles: an admin or
teacher sees only the Admin section, a student only the Student section (a user
holding both sees both). Every page is registered on *every* run regardless —
``visibility`` only controls the sidebar — because a page left out of the
navigation dict stops resolving, which would break both its URL and any
``st.switch_page`` pointing at it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agentic_learning_portal.auth import (
    ADMIN_ROLES,
    current_user,
    flush_remember_me,
    render_sidebar_user,
)

# Load API keys (e.g. GOOGLE_API_KEY) from .env before any page runs.
load_dotenv()

PAGES_DIR = Path(__file__).parent / "views"

st.set_page_config(page_title="Agentic Learning Portal", page_icon="🎓", layout="wide")

# Hand the browser any remember-me cookie write/clear queued by the previous run.
# This has to happen here rather than only inside ``require_roles``: the sign-in
# page is public, so no guard runs for a signed-out visitor — which is precisely
# when a "Remember me" cookie is queued.
flush_remember_me()

user = current_user()
roles = set(user.roles) if user is not None else set()
is_admin = bool(roles & set(ADMIN_ROLES))
# Students, and any signed-in user without an admin role: the student page's own
# guard is ``require_roles()`` (any authenticated user), so the nav matches it.
sees_student = user is not None and ("student" in roles or not is_admin)


def _page(
    module: str,
    title: str,
    icon: str,
    url_path: str,
    *,
    visible: bool,
    default: bool = False,
) -> st.Page:
    """Build a page whose sidebar entry shows only when ``visible``."""
    return st.Page(
        str(PAGES_DIR / module),
        title=title,
        icon=icon,
        url_path=url_path,
        visibility="visible" if visible else "hidden",
        default=default,
    )


# ``default=True`` on the sign-in page is what keeps ``/`` working while leaving
# every other page its named URL. Streamlit allows exactly one default page, so
# it must not depend on the visitor's roles. Signed in, the page still resolves
# (``st.switch_page`` from ``views/signin.py`` lands on the role's home) but it
# drops out of the sidebar, where a Sign in entry would be noise.
signin = _page("signin.py", "Sign in", "🔑", "signin", visible=user is None, default=True)
admin = _page("admin.py", "Admin", "🎓", "admin", visible=is_admin)
assignments = _page(
    "admin/01_assignments.py", "Assignments", "🎓", "assignments", visible=is_admin
)
users = _page("admin/03_users.py", "Users", "👥", "users", visible=is_admin)
student = _page("student.py", "Student", "🎒", "student", visible=sees_student)

# Reached only through the pages above (via ``st.switch_page``), never from the
# sidebar. ``verify`` is public like sign-in: it is opened from an emailed link
# by someone who has no account access yet.
assignment_editor = _page(
    "admin/02_assignment_editor.py",
    "Assignment Editor",
    "✏️",
    "assignment_editor",
    visible=False,
)
take_assignment = _page(
    "student_take.py", "Take assignment", "📝", "take_assignment", visible=False
)
verify = _page("verify.py", "Verify account", "🔐", "verify", visible=False)

# Every page is registered on *every* run; ``visibility`` only controls the
# sidebar. A page left out of the navigation stops resolving, so its URL would
# stop working and any ``st.switch_page`` aimed at it would raise for whoever
# happens to be signed in. The pages gate themselves (``auth.require_roles``),
# so registering one is not a grant — and the hidden ones ride along in the
# section that is showing, so no section header is ever left empty.
all_pages = [
    signin,
    admin,
    assignments,
    users,
    student,
    assignment_editor,
    take_assignment,
    verify,
]

if user is None:
    navigation = {"Account": all_pages}
elif is_admin and sees_student:
    # Holds an admin role *and* ``student``: both sections, admin first, the
    # same precedence ``auth.landing_page_path`` uses for the landing page.
    navigation = {
        "Admin": [page for page in all_pages if page is not student],
        "Student": [student],
    }
elif is_admin:
    navigation = {"Admin": all_pages}
else:
    navigation = {"Student": all_pages}

# Show the signed-in user + Log out in the sidebar on every page. Pages guard
# themselves (see ``auth.require_roles``), so this is cosmetic, not a gate.
render_sidebar_user()

pg = st.navigation(navigation)
pg.run()


def main() -> None:
    """Launch the portal with Streamlit's CLI."""
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    from streamlit.web import cli

    cli.main()
