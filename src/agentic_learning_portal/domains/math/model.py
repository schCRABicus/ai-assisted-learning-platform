from typing import Literal

from pydantic import Field

from agentic_learning_portal.api.model import ProblemGenerationPromptInput


class MathProblemGenerationPromptInput(ProblemGenerationPromptInput):
    """Mathematical problem prompt input."""

    topic: str = Field(
        ...,
        description="The mathematical topic (e.g., 'Algebra', 'Arithmetic', 'Geometry').",
    )
    subtopics: list[str] = Field(
        ...,
        description=(
            "Specific areas to focus on within the topic "
            "(e.g., ['multiplication', 'subtraction'])."
        ),
    )

    context: str = Field(
        ...,
        description="Thematic setting for the problem (e.g., 'Lego', 'Star Wars').",
    )
    complexity: Literal["easy", "medium", "hard"] = Field(
        ...,
        description="The difficulty level of the problem.",
    )
    grade: int = Field(
        ...,
        le=11,
        ge=1,
        description="Grade.",
    )

    def build_user_prompt(self) -> str:
        subtopics = ", ".join(self.subtopics)
        return (
            f"Generate a {self.topic} math problem of {self.complexity} complexity for grade "
            f"{self.grade} student with {subtopics}. Set the problem in a "
            f"{self.context} context. Include a step-by-step solution. "
            f"Keep it structured."
        )
