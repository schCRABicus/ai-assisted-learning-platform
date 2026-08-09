# pydantic-sandbox

A sandbox project for experimenting with **LLMs** and **Streamlit** by building **context-aware math problems**.

The idea is to prompt a language model with user context — grade level, topic, subtopics (e.g. multiplication, subtraction), thematic setting, and difficulty — and receive a structured math problem back. [Pydantic](https://docs.pydantic.dev/) enforces the shape of that output, and [pydantic-ai](https://ai.pydantic.dev/) wires the model call so responses map cleanly onto typed Python objects. A Streamlit front-end (planned) will let you tweak context in the browser and see generated problems interactively.

## Run

**Prerequisites:** Python 3.14+, [uv](https://docs.astral.sh/uv/), and a Google Gemini API key.

```bash
# Clone and enter the project
cd pydantic-sandbox

# Install dependencies
uv sync

# Add your API key (used by pydantic-ai / Gemini)
echo 'GOOGLE_API_KEY=your-key-here' > .env

# Generate a context-aware math problem via LLM
uv run run-app
```

**Streamlit UI (planned):** once a Streamlit app is added, start it with:

```bash
uv run streamlit run app.py
```

## Tests

```bash
uv run pytest
```

`pytest` runs in `asyncio_mode = "auto"` (configured in `pyproject.toml`), so async tests need no explicit marker. Tests mock `Generator._call_llm` to avoid real model calls.

## Architecture

The project is split into a generic **API layer** and domain-specific implementations that plug into it.

### API layer (`api/`)

- `model.py` — `ProblemGenerationPromptInput` (abstract base; subclasses implement `build_user_prompt()`) and `GeneratedTask` (base output schema).
- `generator.py` — `Generator` calls a `pydantic_ai` `Agent` and validates the response against the expected output class. If validation fails, the response and error are folded into a corrective retry prompt and re-sent (up to `n_retry`, default 5) before raising.

### Domain layer (`domains/math/`)

- `model.py` — `MathProblemGenerationPromptInput` adds domain fields (`topic`, `subtopics`, `context`, `complexity`, `grade`) and builds the user prompt; `MathProblem` is the typed LLM output.
- `generator.py` — `MathProblemGenerator` pins the output type to `MathProblem`.
- `__init__.py` — re-exports the domain's public API.

### Generation flow

`MathProblemGenerator.generate(prompt_input)` → `prompt_input.build_user_prompt()` → LLM call → response validated as `MathProblem`. The default model is `google:gemini-3.5-flash`, overridable via `Generator(model=...)`.

### Adding a new domain

Follow the math pattern: subclass `ProblemGenerationPromptInput` (implement `build_user_prompt()`), define an output model, and subclass `Generator` to pin the output type.

## Project layout

```
src/agentic_learning_portal/
├── main.py                  # CLI entry point (run-app)
├── api/
│   ├── generator.py         # LLM call + validation feedback loop
│   └── model.py             # Generic prompt input ABC + base output
└── domains/
    └── math/
        ├── generator.py     # MathProblemGenerator
        ├── model.py         # MathProblemGenerationPromptInput, MathProblem
        └── __init__.py      # Public API re-exports
tests/
├── api/
│   └── test_generator.py
└── domains/
    └── math/
        └── test_math_generator.py
```

## Example output shape

```json
{
  "topic": "Lego",
  "text": "You have 3 boxes with 4 bricks each. How many bricks do you have?",
  "complexity": "easy",
  "correct_answer": 12
}
```

## Tech stack

- **Python 3.14** — managed with **uv**
- **Pydantic v2** — structured data and validation
- **pydantic-ai** — LLM integration with typed outputs
- **Google Gemini** — default model (`google:gemini-3.5-flash`)
- **Streamlit** — planned UI for context inputs and live problem generation
