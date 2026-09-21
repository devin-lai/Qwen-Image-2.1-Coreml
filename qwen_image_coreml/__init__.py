"""Qwen-Image-2.1 as Apple Core ML models.

    from qwen_image_coreml import generate, load_prompt_embeds

    embeds, prompt = load_prompt_embeds("assets/prompts/neon_sign.npz")
    result = generate(embeds, steps=40, seed=42)
    result.image.save("out.png")
"""

from .pipeline import DEFAULT_MODELS, Result, generate, load_prompt_embeds
from .scheduler import FlowMatchEulerScheduler

__all__ = [
    "DEFAULT_MODELS",
    "FlowMatchEulerScheduler",
    "Result",
    "generate",
    "load_prompt_embeds",
]
__version__ = "0.1.0"
