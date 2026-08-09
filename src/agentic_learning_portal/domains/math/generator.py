from agentic_learning_portal.api.generator import DEFAULT_SYSTEM_PROMPT, Generator
from agentic_learning_portal.domains.math.model import MathProblem, MathProblemGenerationPromptInput


class MathProblemGenerator(Generator):
    """Math Problem Generator class."""

    async def generate(
        self,
        prompt_input: MathProblemGenerationPromptInput,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> MathProblem:
        return await super().generate(
            prompt_input,
            MathProblem,
            system_prompt=system_prompt,
        )
