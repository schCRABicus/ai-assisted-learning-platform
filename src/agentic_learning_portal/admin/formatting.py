"""Display helpers for rendering portal content as markdown.

The generation model is instructed to write math in plain text, but it
occasionally emits LaTeX markup (``$$...$$``, ``\\text{...}``, ``\\frac{}{}``,
or a stray backslash like ``\\558``) anyway. Streamlit's markdown does not
render LaTeX, so :func:`latex_to_plain_text` converts the common constructs into
readable plain text as a display-time fallback.

Student-typed text (an answer or a worked solution) needs the opposite
treatment: it is not LaTeX, but markdown would still collapse its line breaks.
:func:`literal_lines` keeps the shape the student typed.
"""

from __future__ import annotations

import re

# Math delimiters: $$...$$, \(...\), \[...\], and escaped dollars.
_MATH_DELIMITERS = re.compile(r"\$\$|\\\$|\\\(|\\\)|\\\[|\\\]")

# \text{...}, \mathrm{...}, \operatorname{...} -> contents.
_NAMED_TEXT = re.compile(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}")

# \frac{a}{b} -> a/b, applied repeatedly so nested fractions resolve.
_FLAT_FRAC = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")

# \sqrt{a} -> √a.
_SQRT = re.compile(r"\\sqrt\{([^{}]*)\}")

# \left( \right) / \left[ \right] -> plain brackets.
_LEFT_RIGHT = [
    (r"\left(", "("),
    (r"\right)", ")"),
    (r"\left[", "["),
    (r"\right]", "]"),
]

# Common operator commands -> unicode. Applied longest-first so a command that
# is a prefix of another (e.g. \ne vs \neq) is replaced correctly.
_OPERATORS = {
    "\\times": "×",
    "\\cdot": "·",
    "\\div": "÷",
    "\\pm": "±",
    "\\le": "≤",
    "\\ge": "≥",
    "\\neq": "≠",
    "\\ne": "≠",
    "\\approx": "≈",
    "\\infty": "∞",
}


def latex_to_plain_text(text: str) -> str:
    """Best-effort conversion of LaTeX math markup into readable plain text."""
    if not text:
        return text

    text = _MATH_DELIMITERS.sub("", text)
    text = _NAMED_TEXT.sub(r"\1", text)

    # Resolve nested fractions one level at a time.
    previous: str | None = None
    while previous != text:
        previous = text
        text = _FLAT_FRAC.sub(r"\1/\2", text)

    text = _SQRT.sub(r"√\1", text)
    for pattern, replacement in _LEFT_RIGHT:
        text = text.replace(pattern, replacement)
    for command in sorted(_OPERATORS, key=len, reverse=True):
        text = text.replace(command, _OPERATORS[command])

    # Drop any remaining backslashes (e.g. the malformed "\558") and stray braces.
    text = text.replace("\\", "").replace("{", "").replace("}", "")

    return text.strip()


def literal_lines(text: str) -> str:
    """Return student-typed text with its line breaks preserved in markdown.

    Markdown folds a single newline into a space, which would run a student's
    step-by-step working together into one paragraph. Two trailing spaces are
    markdown's hard break, so a multi-step solution keeps the shape it was typed
    in. The text is otherwise passed through untouched — unlike generated
    content it is not LaTeX, so it is not run through
    :func:`latex_to_plain_text`.
    """
    return text.replace("\n", "  \n")
