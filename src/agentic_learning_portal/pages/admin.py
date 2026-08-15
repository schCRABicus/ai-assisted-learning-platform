"""Admin endpoint of the Agentic Learning Portal.

Run the whole portal with ``uv run run-portal``; this page is served at
``/admin``.
"""

from __future__ import annotations

import asyncio
import threading
import time

import streamlit as st

from agentic_learning_portal.admin.formatting import latex_to_plain_text
from agentic_learning_portal.admin.subtopic_suggester import suggest_subtopics
from agentic_learning_portal.api.progress import CollectingProgressListener
from agentic_learning_portal.auth import require_roles
from agentic_learning_portal.domains.math import (
    MathProblemGenerationPromptInput,
    MathProblemGenerator,
    build_judge_ensemble,
)

# Gate the page before any widget (the topic input below fires an LLM call for
# subtopic suggestions, so it must never run for unauthenticated visitors).
require_roles("admin", "teacher")

st.title("🎓 Task Generation Admin")

DEFAULT_CONTEXT = "Everyday life"
DEFAULT_MODEL = "google:gemini-3.5-flash"
POLL_INTERVAL_S = 0.25

# Icon per progress stage, shown in the live status panel while a task is generated.
_EVENT_ICONS = {
    "generate": "📝",
    "validate": "✅",
    "verify": "🔎",
    "retry": "🔁",
    "done": "🎉",
    "error": "❌",
}


def _event_icon(stage: str) -> str:
    return _EVENT_ICONS.get(stage, "•")


def _run_generation_in_thread(
    prompt_input: MathProblemGenerationPromptInput,
    gen_state: dict,
) -> None:
    """Generate a task in a daemon thread, writing progress + outcome to ``gen_state``.

    ``gen_state`` is a plain dict shared with the main script: the worker
    appends ``ProgressEvent`` objects to ``gen_state["log"]`` and stores the
    finished ``GeneratedTask`` in ``gen_state["result"]`` (or an error string in
    ``gen_state["error"]``). The main script polls it on every rerun and renders
    the accumulated log. Nothing here touches ``st.session_state``.
    """
    listener = CollectingProgressListener(gen_state["log"])

    async def _run() -> None:
        try:
            task = await MathProblemGenerator().generate(
                prompt_input,
                judge=build_judge_ensemble() or None,
                listener=listener,
            )
            gen_state["result"] = task
        except RuntimeError as e:
            gen_state["error"] = str(e)
        except Exception as e:  # noqa: BLE001 - surface any failure in the UI
            gen_state["error"] = f"{type(e).__name__}: {e}"

    asyncio.run(_run())


def _run_suggest_subtopics(topic: str, model: str) -> list[str]:
    return asyncio.run(suggest_subtopics(topic, model=model))


def _reset_suggestion_state(subtopics: list[str], topic: str) -> None:
    """Store the current suggestion set, leaving the selection empty.

    The suggestions populate the multiselect's options but nothing is
    pre-selected, so the admin chooses the subtopics to focus on.
    """
    st.session_state["suggested_subtopics"] = list(subtopics)
    st.session_state["suggested_for"] = topic
    st.session_state["selected_subtopics"] = []


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
# Generation runs in a background thread so the page can show live progress.
# ``generating`` is the in-flight flag, ``gen_state`` is a plain dict shared
# with the worker thread (log events + final outcome), and ``generation_status``
# carries a one-shot success/error message back.
if "generating" not in st.session_state:
    st.session_state["generating"] = False
if "gen_state" not in st.session_state:
    st.session_state["gen_state"] = None
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
elif not selected_subtopics:
    st.caption("💡 Select the subtopics to focus on (or add your own below).")

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
    # Freeze the inputs, clear the previous result, and start generation in a
    # background thread; then rerun so the button renders disabled while the
    # live progress panel below streams the worker's events in one run.
    prompt_input = MathProblemGenerationPromptInput(
        topic=topic.strip(),
        subtopics=subtopics,
        context=context.strip() or DEFAULT_CONTEXT,
        complexity=complexity,
        grade=grade,
    )
    st.session_state["last_task"] = None
    st.session_state["last_prompt"] = prompt_input
    st.session_state["gen_state"] = {"log": [], "result": None, "error": None}
    st.session_state["generating"] = True
    threading.Thread(
        target=_run_generation_in_thread,
        args=(prompt_input, st.session_state["gen_state"]),
        daemon=True,
    ).start()
    st.rerun()

if st.session_state["generating"]:
    gen_state = st.session_state["gen_state"]
    status = st.status("✨ Generating the task...", expanded=True)

    # Live progress in a single script run: append each new progress line once
    # and never re-render previous ones. Streamlit streams these deltas to the
    # browser as they're created, so the panel grows in place while the rest of
    # the page stays untouched — no per-poll full rerun, no jumping. The worker
    # thread appends to ``gen_state["log"]``; this loop drains the tail until
    # the run finishes (the worker writes the result only after all events).
    last_rendered = 0
    while gen_state["result"] is None and gen_state["error"] is None:
        log = gen_state["log"]
        new_events = log[last_rendered:]
        if new_events:
            for event in new_events:
                st.markdown(f"{_event_icon(event.stage)} {event.message}")
            last_rendered = len(log)
        time.sleep(POLL_INTERVAL_S)

    # Drain anything appended since the last pass, then fold the outcome in.
    for event in gen_state["log"][last_rendered:]:
        st.markdown(f"{_event_icon(event.stage)} {event.message}")

    if gen_state["error"] is not None:
        status.update(label="❌ Task generation failed", state="error")
        st.session_state["generation_status"] = (
            "error",
            f"Could not generate a validated task: {gen_state['error']}",
        )
    else:
        status.update(label="✅ Task generated and verified", state="complete")
        st.session_state["last_task"] = gen_state["result"]
        st.session_state["generation_status"] = (
            "success",
            "Task generated and schema-validated.",
        )
    st.session_state["generating"] = False
    # One final rerun to re-enable the button and render status + result.
    st.rerun()

status = st.session_state.pop("generation_status", None)
if status is not None:
    kind, message = status
    if kind == "success":
        st.success(message)
    else:
        st.error(message)

# After a run finishes, keep the step history visible (collapsed) so the admin
# can review exactly what the pipeline did.
if st.session_state["gen_state"] is not None and not st.session_state["generating"]:
    last_log = st.session_state["gen_state"]["log"]
    with st.expander(f"📜 Generation steps ({len(last_log)})", expanded=False):
        for event in last_log:
            st.markdown(f"{_event_icon(event.stage)} {event.message}")

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