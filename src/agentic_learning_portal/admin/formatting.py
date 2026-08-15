"""Display helpers for rendering LLM-generated content in the portal.

The generation model is instructed to write math in plain text, but it
occasionally emits LaTeX markup (``$$...$$``, ``\\text{...}``, ``\\frac{}{}``,
or a stray backslash like ``\\558``) anyway. Streamlit's markdown does not
render LaTeX, so this converts the common constructs into readable plain text
as a display-time fallback.
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