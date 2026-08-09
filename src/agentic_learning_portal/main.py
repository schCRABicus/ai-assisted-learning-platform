import asyncio
import logging
import os

from dotenv import load_dotenv

from .domains.math import MathProblemGenerationPromptInput, MathProblemGenerator, WolframAlphaJudge

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
        grade=3,
    )

    print(f"Asking LLM to generate a math problem for grade {input.grade} student with {' and '.join(subtopics)}...")
    judge = WolframAlphaJudge() if os.environ.get("WOLFRAM_APP_ID") else None
    if judge is None:
        print("WOLFRAM_APP_ID not set — skipping Wolfram|Alpha verification.")

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