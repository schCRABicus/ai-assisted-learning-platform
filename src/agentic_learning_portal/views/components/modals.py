import asyncio

import streamlit as st

from agentic_learning_portal.admin.subtopic_suggester import suggest_subtopics
from agentic_learning_portal.storage import Storage
from agentic_learning_portal.storage.factory import StorageFactory
from agentic_learning_portal.api.llm import MODELS, ModelChain
from agentic_learning_portal.domains.math import MathProblemGenerationPromptInput

STORAGE_FACTORY = StorageFactory()

DEFAULT_TASK_CONTEXT = "Plants vs Zombie"
DEFAULT_SUBTOPIC_SUGGESTION_MODEL = MODELS["subtopic_suggestion"]


def get_storage() -> Storage:
    """Return the :class:`Storage` the portal should use.

    The backend is chosen by ``STORAGE_BACKEND`` (``"sqlite"`` or ``"memory"``),
    read at call time so tests and ``.env`` can switch it after import. The
    sqlite backend keeps one instance per thread (``seed_admin_from_env`` runs
    on construction, so the ``.env``-configured admin is available from the
    start); the memory backend is a single shared instance.
    """
    return STORAGE_FACTORY.get_storage()


# Define the confirmation pop-up dialog
@st.dialog("Confirm Assignment Deletion", dismissible=True)
def delete_assignment_dialog(assignment_id: int) -> None:
    st.write("Are you sure you want to delete this assignment? This cannot be undone.")

    col1, col2 = st.columns(2)

    if col1.button("Yes, Delete", type="primary"):
      get_storage().delete_assignment(assignment_id)
      st.success("Item deleted successfully!")
      # st.session_state.show_dialog = False
      st.rerun()

    if col2.button("Cancel"):
      # st.session_state.show_dialog = False
      st.rerun()


@st.cache
def get_subtopic_suggestions(topic: str, model: ModelChain) -> list[str]:
    return asyncio.run(suggest_subtopics(topic, model=model))

@st.dialog("Task Creation", dismissible=True)
def create_task_dialog(assignment_id: int) -> None:
    st.caption("Describe the task below, then Generate.")

    # --- topic + dynamically suggested subtopics -----------------------------------

    topic = st.text_input(
      "📚 Topic",
      placeholder="e.g. Arithmetic, Algebra, Geometry",
      help="The math topic to generate a task for.",
      # key=f"topic_for_assigment_{assignment_id}",
    )

    # if topic.strip() and topic.strip().lower() != st.session_state.get(suggested_for_key, "").lower():
    if topic.strip():
      with st.spinner("🔍 Suggesting subtopics..."):
        suggestions = get_subtopic_suggestions(topic, DEFAULT_SUBTOPIC_SUGGESTION_MODEL)
      # st.session_state[suggested_subtopics_key] = list(suggestions)
      # st.session_state[suggested_for_key] = topic.strip()
      # st.session_state[selected_subtopics_key] = []
      if not suggestions:
        st.warning(
          "The LLM could not suggest subtopics for this topic. Add them "
          "manually below."
        )

    # suggested = st.session_state.get(suggested_subtopics_key, [])
    selected_subtopics = st.multiselect(
      "🧩 Subtopics",
      # options=suggested,
      options=suggestions,
      # key=selected_subtopics_key,
      help="LLM-suggested subtopics; select the ones to focus on.",
    )
    manual = st.text_input(
      "✍️ Add your own subtopics (comma-separated)",
      placeholder="e.g. fractions, word problems",
      # key=f"carousel_manual_subtopics_{index}",
    )
    manual_subtopics = [s.strip() for s in manual.split(",") if s.strip()]
    subtopics = list(selected_subtopics) + [
      s for s in manual_subtopics if s not in selected_subtopics
    ]

    # if not suggested:
    if not suggestions:
      st.caption("💡 Enter a topic to get LLM-suggested subtopics.")
    elif not selected_subtopics:
      st.caption("💡 Select the subtopics to focus on (or add your own below).")

    # --- other parameters ----------------------------------------------------------

    context = st.text_input(
      "🎭 Context",
      value=DEFAULT_TASK_CONTEXT,
      # key=f"carousel_context_{index}",
      help="Thematic setting for the problem (e.g. Lego, Plants vs Zombies).",
    )

    col_grade, col_complexity = st.columns(2)
    grade = col_grade.selectbox(
      "🎓 Grade",
      options=list(range(1, 12)),
      index=4,
      # key=f"carousel_grade_{index}",
      help="Student grade level (1-11).",
    )
    complexity = col_complexity.selectbox(
      "📊 Complexity",
      options=["easy", "medium", "hard"],
      index=0,
      # key=f"carousel_complexity_{index}",
    )

    # --- generate / discard ---------------------------------------------------------

    generate_clicked = st.button(
      "🎯 Generate",
      type="primary",
      disabled=not (topic.strip() and subtopics),
      # key=f"carousel_generate_{index}",
    )

    if generate_clicked:
      prompt_input = MathProblemGenerationPromptInput(
        topic=topic.strip(),
        subtopics=subtopics,
        context=context.strip() or DEFAULT_TASK_CONTEXT,
        complexity=complexity,
        grade=grade,
      )
      # st.session_state[f"carousel_last_prompt_{index}"] = prompt_input
      # st.session_state[f"carousel_gen_state_{index}"] = {
      #   "log": [],
      #   "result": None,
      #   "error": None,
      # }
      # st.session_state[f"carousel_generating_{index}"] = True
      # threading.Thread(
      #   target=_run_generation_in_thread,
      #   args=(
      #     prompt_input,
      #     st.session_state[f"carousel_gen_state_{index}"],
      #   ),
      #   daemon=True,
      # ).start()
      # st.rerun()
