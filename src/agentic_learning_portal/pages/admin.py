"""Admin endpoint of the Agentic Learning Portal.

Run the whole portal with ``uv run run-portal``; this page is served at
``/admin``.
"""

from __future__ import annotations

import asyncio

import streamlit as st

from agentic_learning_portal.admin.formatting import latex_to_plain_text
from agentic_learning_portal.admin.subtopic_suggester import suggest_subtopics
from agentic_learning_portal.domains.math import (
    MathProblemGenerationPromptInput,
    MathProblemGenerator,
)

st.title("🎓 Task Generation Admin")

DEFAULT_CONTEXT = "Everyday life"
DEFAULT_MODEL = "google:gemini-3.5-flash"


def _run_suggest_subtopics(topic: str, model: str) -> list[str]:
    return asyncio.run(suggest_subtopics(topic, model=model))


def _reset_suggestion_state(subtopics: list[str], topic: str) -> None:
    """Store the current suggestion set and select all of it by default."""
    st.session_state["suggested_subtopics"] = list(subtopics)
    st.session_state["suggested_for"] = topic
    st.session_state["selected_subtopics"] = list(subtopics)


# --- persistent state ---------------------------------------------------------
if "suggested_subtopics" not in st.session_state:
    st.session_state["suggested_subtopics"] = []
if "suggested_for" not in st.session_state:
    st.session_state["suggested_for"] = ""
if "selected_subtopics" not in st.session_state:
    st.session_state["selected_subtopics"] = []
if "last_task" not in st.session_state:
    st.session_state["last_task"] = None
if "last_prompt" not in st.session_state:
    st.session_state["last_prompt"] = None
# Generation is split across reruns so the button can stay disabled for the
# whole duration of the (blocking) LLM call. ``generating`` is the in-flight
# flag, ``pending_prompt`` carries the frozen inputs across the rerun, and
# ``generation_status`` carries a one-shot success/error message back.
if "generating" not in st.session_state:
    st.session_state["generating"] = False
if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None
if "generation_status" not in st.session_state:
    st.session_state["generation_status"] = None

# --- topic + dynamically suggested subtopics --------------------------------

topic = st.text_input(
    "📚 Topic",
    placeholder="e.g. Arithmetic, Algebra, Geometry",
    help="The math topic to generate a task for.",
)

# Suggest subtopics as soon as the topic is entered, and only once per topic:
# ``suggested_for`` records the topic the current suggestions were generated
# for, so a rerun caused by any other widget does not fire the LLM call again.
if topic.strip() and topic.strip().lower() != st.session_state["suggested_for"].lower():
    with st.spinner("🔍 Suggesting subtopics..."):
        suggestions = _run_suggest_subtopics(topic, DEFAULT_MODEL)
    if suggestions:
        _reset_suggestion_state(suggestions, topic.strip())
    else:
        # Record the topic even on failure so we don't retry on every rerun.
        _reset_suggestion_state([], topic.strip())
        st.warning(
            "The LLM could not suggest subtopics for this topic. Add them "
            "manually below."
        )

suggested = st.session_state["suggested_subtopics"]
selected_subtopics = st.multiselect(
    "🧩 Subtopics",
    options=suggested,
    key="selected_subtopics",
    help="LLM-suggested subtopics; select the ones to focus on.",
)
manual = st.text_input(
    "✍️ Add your own subtopics (comma-separated)",
    placeholder="e.g. fractions, word problems",
)
manual_subtopics = [s.strip() for s in manual.split(",") if s.strip()]
subtopics = list(selected_subtopics) + [
    s for s in manual_subtopics if s not in selected_subtopics
]

if not suggested:
    st.caption("💡 Enter a topic to get LLM-suggested subtopics.")

# --- other parameters ---------------------------------------------------------

context = st.text_input(
    "🎭 Context",
    value=DEFAULT_CONTEXT,
    help="Thematic setting for the problem (e.g. Lego, Plants vs Zombies).",
)

col_grade, col_complexity = st.columns(2)
grade = col_grade.selectbox(
    "🎓 Grade",
    options=list(range(1, 12)),
    index=4,
    help="Student grade level (1-11).",
)
complexity = col_complexity.selectbox(
    "📊 Complexity",
    options=["easy", "medium", "hard"],
    index=0,
)

# --- generate -----------------------------------------------------------------

generate_clicked = st.button(
    "Generate task",
    icon="✨",
    type="primary",
    disabled=st.session_state["generating"] or not (topic.strip() and subtopics),
)

if generate_clicked:
    # Freeze the inputs and mark generation as in-flight, then rerun so the
    # button renders disabled for the whole blocking call below.
    st.session_state["generating"] = True
    st.session_state["pending_prompt"] = MathProblemGenerationPromptInput(
        topic=topic.strip(),
        subtopics=subtopics,
        context=context.strip() or DEFAULT_CONTEXT,
        complexity=complexity,
        grade=grade,
    )
    st.rerun()

if st.session_state["generating"] and st.session_state["pending_prompt"] is not None:
    prompt_input = st.session_state.pop("pending_prompt")
    try:
        with st.spinner("⏳ Generating the task..."):
            task = asyncio.run(MathProblemGenerator().generate(prompt_input))
    except RuntimeError as e:
        st.session_state["generation_status"] = (
            "error",
            f"Could not generate a validated task: {e}",
        )
    else:
        st.session_state["last_task"] = task
        st.session_state["last_prompt"] = prompt_input
        st.session_state["generation_status"] = (
            "success",
            "Task generated and schema-validated.",
        )
    st.session_state["generating"] = False
    # Rerun to re-enable the button and render the one-shot status + result.
    st.rerun()

status = st.session_state.pop("generation_status", None)
if status is not None:
    kind, message = status
    if kind == "success":
        st.success(message)
    else:
        st.error(message)

# --- result -------------------------------------------------------------------

task = st.session_state["last_task"]
if task is not None:
    prompt_input = st.session_state["last_prompt"]

    st.divider()
    st.subheader("Generated task")

    meta = st.columns(4)
    meta[0].markdown(f"**📚 Topic:** {task.topic}")
    meta[1].markdown(f"**🎓 Grade:** {prompt_input.grade}")
    meta[2].markdown(f"**📊 Complexity:** {task.complexity}")
    meta[3].markdown(f"**🧩 Subtopics:** {', '.join(prompt_input.subtopics)}")

    st.markdown("---")
    st.markdown(f"**📝 Problem**\n\n{latex_to_plain_text(task.text)}")
    st.markdown(f"**✅ Correct answer:** `{task.correct_answer}`")
    with st.expander("💡 Solution", expanded=False):
        st.markdown(latex_to_plain_text(task.solution))