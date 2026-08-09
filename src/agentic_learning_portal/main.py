from .domains.math import MathProblem, MathProblemGenerationPromptInput, MathProblemGenerator
from pydantic_ai import Agent
from dotenv import load_dotenv
import asyncio

load_dotenv()


async def generate_math_problem():
    subtopics = ["multiplication", "subtraction", "division"]
    input = MathProblemGenerationPromptInput(
        topic="Arithmetic",
        subtopics=subtopics,
        context="Plants vs Zombies",
        complexity="hard",
        grade=3,
    )

    print(f"Asking LLM to generate a math problem for grade {input.grade} student with {' and '.join(subtopics)}...")
    task = await MathProblemGenerator().generate(
        prompt_input=input,
    )

    print("\n--- LLM Response ---")
    print(task)


def main():
    asyncio.run(generate_math_problem())


if __name__ == '__main__':
    main()
