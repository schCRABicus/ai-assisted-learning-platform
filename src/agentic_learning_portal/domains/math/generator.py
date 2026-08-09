from agentic_learning_portal.api.generator import DEFAULT_SYSTEM_PROMPT, Generator
from agentic_learning_portal.api.model import GeneratedTask
from agentic_learning_portal.domains.math.model import MathProblemGenerationPromptInput


class MathProblemGenerator(Generator):
    """Math Problem Generator class."""

    async def generate(
        self,
        prompt_input: MathProblemGenerationPromptInput,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> GeneratedTask:
        return await super().generate(
            prompt_input,
            GeneratedTask,
            system_prompt=system_prompt,
        )
