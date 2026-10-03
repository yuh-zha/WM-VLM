"""Hugging Face configuration for WM-VLM checkpoints."""

from __future__ import annotations

from typing import Any

from transformers import Qwen2_5_VLConfig


class WMVLMConfig(Qwen2_5_VLConfig):
    """Qwen2.5-VL configuration with a distinct AutoClass model type."""

    model_type = "wm_vlm"

    def __getattribute__(self, name: str) -> Any:
        """Keep the composite model type stable across Transformers releases.

        Transformers 4.57.1 forwards most top-level Qwen2.5-VL config
        attributes to ``text_config``.  Unlike newer 4.57.x releases, that
        forwarding also includes ``model_type``, which makes a native WM-VLM
        config appear to be ``qwen2_5_vl_text`` after loading.  Read this one
        class discriminator directly from the WM-VLM class while preserving
        Qwen's forwarding behavior for every other attribute.
        """
        if name == "model_type":
            return type(self).model_type
        return super().__getattribute__(name)

    @classmethod
    def from_qwen_config(cls, config: Qwen2_5_VLConfig) -> "WMVLMConfig":
        """Create a WM-VLM training config from a base Qwen config."""
        if isinstance(config, cls):
            config.architectures = ["WMVLMForConditionalGeneration"]
            return config
        payload: dict[str, Any] = config.to_dict()
        payload.pop("model_type", None)
        payload["architectures"] = ["WMVLMForConditionalGeneration"]
        return cls(**payload)
