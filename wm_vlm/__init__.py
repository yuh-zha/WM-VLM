"""WM-VLM visual reasoning for Qwen2.5-VL."""

from transformers import AutoConfig, AutoModelForImageTextToText

from .configuration_wm_vlm import WMVLMConfig
from .modeling import (
    WMVLMForConditionalGeneration,
    WMVLMOutput,
    build_parallel_reasoning_attention_masks,
    build_vlm_only_attention_masks,
)
from .constants import (
    BOTTOM_BRANCH_ONLY_VARIANT,
    BRANCH_ONLY_VARIANT,
    MIDDLE_BRANCH_ONLY_VARIANT,
)
from .generation import integrate_flow_tokens
from .inference import generate_wm_vlm_answer
from .tokens import configure_wm_vlm_model


def register_wm_vlm_auto_classes() -> None:
    """Register installed-package and Hub dynamic-code AutoClass mappings."""
    AutoConfig.register(WMVLMConfig.model_type, WMVLMConfig, exist_ok=True)
    AutoModelForImageTextToText.register(
        WMVLMConfig,
        WMVLMForConditionalGeneration,
        exist_ok=True,
    )
    WMVLMConfig.register_for_auto_class()
    WMVLMForConditionalGeneration.register_for_auto_class(
        "AutoModelForImageTextToText"
    )


register_wm_vlm_auto_classes()

__all__ = [
    "WMVLMConfig",
    "WMVLMForConditionalGeneration",
    "WMVLMOutput",
    "build_parallel_reasoning_attention_masks",
    "build_vlm_only_attention_masks",
    "BOTTOM_BRANCH_ONLY_VARIANT",
    "BRANCH_ONLY_VARIANT",
    "MIDDLE_BRANCH_ONLY_VARIANT",
    "configure_wm_vlm_model",
    "generate_wm_vlm_answer",
    "integrate_flow_tokens",
    "register_wm_vlm_auto_classes",
]
