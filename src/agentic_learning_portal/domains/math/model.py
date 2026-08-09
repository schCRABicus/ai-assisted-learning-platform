from typing import Literal

from pydantic import BaseModel, Field

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
            f"{self.context} context. Keep it structured."
        )


class MathProblem(BaseModel):
    """Pydantic model representing a mathematical problem."""

    topic: str = Field(
        ...,
        description="The mathematical topic (e.g., 'Algebra', 'Calculus', 'Geometry').",
    )
    text: str = Field(
        ...,
        description="The actual text or question of the mathematical problem.",
    )
    complexity: Literal["easy", "medium", "hard"] = Field(
        ...,
        description="The difficulty level of the problem.",
    )
    correct_answer: str | int | float = Field(
        ...,
        description="The correct answer to the problem, which can be an integer, float, or string.",
    )
