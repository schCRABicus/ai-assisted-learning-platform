import asyncio
import logging

from dotenv import load_dotenv

from .domains.math import (
    MathProblemGenerationPromptInput,
    MathProblemGenerator,
    build_judge_ensemble,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

load_dotenv()


async def generate_math_problem():
    subtopics = ["multiplication", "subtraction", "division"]
    input = MathProblemGenerationPromptInput(
        topic="Arithmetic",
        subtopics=subtopics,
        context="Plants vs Zombies",
        complexity="hard",
        grade=5,
    )

    print(f"Asking LLM to generate a math problem for grade {input.grade} student with {' and '.join(subtopics)}...")

    # A list means the task must be verified by every judge (judge-ensemble
    # consensus); empty means no judge keys are set and generation is plain.
    judges = build_judge_ensemble()
    if judges:
        print("Verifying with judge ensemble: " + ", ".join(j.__class__.__name__ for j in judges))
    else:
        print("No judge keys set — skipping verification (plain generation).")
    judge = judges if judges else None

    task = await MathProblemGenerator().generate(
        prompt_input=input,
        judge=judge,
    )

    print("\n--- LLM Response ---")
    print(task)


def main():
    asyncio.run(generate_math_problem())


if __name__ == '__main__':
    main()