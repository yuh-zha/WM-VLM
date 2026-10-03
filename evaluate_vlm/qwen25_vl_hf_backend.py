"""Sequential Hugging Face backend for direct Qwen2.5-VL generation."""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def _torch_dtype(name: str) -> torch.dtype | str:
    normalized = name.lower()
    if normalized == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[normalized]
    except KeyError as error:
        raise ValueError(f"Unsupported Qwen2.5-VL dtype: {name}") from error


class Qwen25VLHFBackend:
    """Load base or fine-tuned weights into the stock Qwen generation path."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 128,
        top_p: float = 1.0,
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        attn_implementation: str = "sdpa",
        image_size: int | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if image_size is not None and image_size < 1:
            raise ValueError("image_size must be positive when provided")

        processor_kwargs = {}
        if image_size is not None:
            pixel_budget = int(image_size) ** 2
            processor_kwargs.update(min_pixels=pixel_budget, max_pixels=pixel_budget)
        self.processor = AutoProcessor.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            use_fast=False,
            **processor_kwargs,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model,
            torch_dtype=_torch_dtype(dtype),
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        top_level_model_type = self.model.config.to_dict().get("model_type")
        if top_level_model_type != "qwen2_5_vl":
            raise ValueError(
                "qwen25_vl_hf requires a Qwen2.5-VL checkpoint; "
                f"found top-level model_type={top_level_model_type!r}"
            )
        self.model.to(torch.device(device))
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompts: list[str], images: list[list[Image.Image]]) -> list[str]:
        predictions = []
        for prompt, image_list in zip(prompts, images, strict=True):
            if not image_list:
                raise ValueError("Qwen2.5-VL direct evaluation requires at least one image")
            content = [{"type": "image"} for _ in image_list]
            content.append({"type": "text", "text": prompt})
            rendered = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            batch = self.processor(
                text=[rendered],
                images=image_list,
                padding=True,
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            generate_kwargs = {
                "max_new_tokens": self.max_tokens,
                "do_sample": self.temperature > 0,
                "use_cache": True,
            }
            if self.temperature > 0:
                generate_kwargs.update(temperature=self.temperature, top_p=self.top_p)
            else:
                generate_kwargs.update(temperature=None, top_p=None)
            generated = self.model.generate(**batch, **generate_kwargs)
            prompt_length = int(batch["input_ids"].shape[1])
            predictions.append(
                self.processor.tokenizer.decode(
                    generated[0, prompt_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
            )
        return predictions
