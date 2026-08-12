import asyncio
import logging
import os

from dotenv import load_dotenv

from .domains.math import (
    MathProblemGenerationPromptInput,
    MathProblemGenerator,
    QwenMathJudge,
    WolframAlphaJudge,
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
        grade=8,
    )

    print(f"Asking LLM to generate a math problem for grade {input.grade} student with {' and '.join(subtopics)}...")
    judges: list = []
    if os.environ.get("WOLFRAM_APP_ID"):
        if os.environ.get("GROQ_API_KEY"):
            # Default judge model (groq:llama-3.3-70b-versatile) is independent of
            # the Gemini generator.
            judges.append(WolframAlphaJudge())
        else:
            print(
                "GROQ_API_KEY not set — using Gemini for the judge's "
                "translation pass (less independent than the default Groq judge)."
            )
            judges.append(WolframAlphaJudge(model="google:gemini-3.5-flash"))
    else:
        print("WOLFRAM_APP_ID not set — skipping Wolfram|Alpha verification.")
    if os.environ.get("GROQ_API_KEY"):
        judges.append(QwenMathJudge())
    else:
        print("GROQ_API_KEY not set — skipping Qwen verification.")

    # A list means the task must be verified by every judge (judge-ensemble consensus).
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