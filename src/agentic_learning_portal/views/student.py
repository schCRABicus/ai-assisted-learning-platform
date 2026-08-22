"""Student endpoint of the Agentic Learning Portal.

Placeholder — served at ``/student``. The student portal (start an assignment,
answer tasks, see results) will live here later.
"""

from __future__ import annotations

import streamlit as st

from agentic_learning_portal.auth import require_roles

# Any authenticated user may enter the student portal; no role required.
user = require_roles()

st.title("🧑‍🎓 Student")
st.caption(f"Signed in as **{user.username}**.")
st.info("The student portal is coming soon.")