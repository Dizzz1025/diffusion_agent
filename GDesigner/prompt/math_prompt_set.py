from .gsm8k_prompt_set import GSM8KPromptSet
from .prompt_set_registry import PromptSetRegistry


@PromptSetRegistry.register("math")
class MathPromptSet(GSM8KPromptSet):
    """Reuse GSM8K-style cooperative math prompts for the MATH dataset."""
    pass
