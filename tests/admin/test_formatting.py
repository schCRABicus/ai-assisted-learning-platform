from __future__ import annotations

from agentic_learning_portal.admin.formatting import latex_to_plain_text


def test_plain_text_passes_through() -> None:
    assert (
        latex_to_plain_text("First multiply, then divide the total.")
        == "First multiply, then divide the total."
    )


def test_strips_display_math_delimiters() -> None:
    assert latex_to_plain_text("$$45 \\times 12 = 540$$") == "45 × 12 = 540"


def test_handles_inline_math_delimiters() -> None:
    assert latex_to_plain_text("Solve \\(3x = 6\\).") == "Solve 3x = 6."


def test_translates_fraction_to_slash() -> None:
    assert latex_to_plain_text("$$\\frac{1}{2}$$") == "1/2"


def test_resolves_nested_fraction() -> None:
    assert latex_to_plain_text("\\frac{1}{\\frac{2}{3}}") == "1/2/3"


def test_extracts_named_text_command() -> None:
    assert latex_to_plain_text("\\text{Cost} = 540 + 18") == "Cost = 540 + 18"


def test_strips_broken_backslash_before_number() -> None:
    # The model sometimes emits a stray backslash before a number ("\\558").
    assert (
        latex_to_plain_text("$$\\text{Cost} = 540 + 18 = \\558$$")
        == "Cost = 540 + 18 = 558"
    )


def test_translates_sqrt_and_operators() -> None:
    assert latex_to_plain_text("\\sqrt{16} = 4") == "√16 = 4"
    assert latex_to_plain_text("a \\cdot b \\div c \\ne d") == "a · b ÷ c ≠ d"


def test_returns_empty_string_unchanged() -> None:
    assert latex_to_plain_text("") == ""