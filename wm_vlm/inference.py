"""Raw-input preparation and text decoding around ``model.generate``."""

from __future__ import annotations

from typing import Any

import torch

from .constants import FULL_VISION_TOKENS_TARGET
from .data import build_user_messages, preprocess_with_qwen_vl_utils


@torch.inference_mode()
def encode_oracle_image_latents(
    model: Any,
    processor: Any,
    oracle_image: Any,
    *,
    image_processor: Any | None = None,
) -> torch.Tensor:
    """Encode a hidden helper image into the checkpoint's visual-token space."""
    helper_image_processor = (
        processor.image_processor if image_processor is None else image_processor
    )
    if bool(
        getattr(
            model.config,
            "wm_vlm_helper_qwen_vl_utils_preprocess",
            getattr(model.config, "wm_vlm_qwen_vl_utils_preprocess", False),
        )
    ):
        oracle_image = preprocess_with_qwen_vl_utils(oracle_image)
    helper_batch = helper_image_processor(
        images=[oracle_image],
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    helper_embeds = model._encode_visual(
        helper_batch["pixel_values"].to(device),
        helper_batch["image_grid_thw"].to(device),
    )
    latent_size = int(model.config.wm_vlm_latent_size)
    target_mode = getattr(model.config, "wm_vlm_target_mode", None)
    if target_mode != FULL_VISION_TOKENS_TARGET:
        raise ValueError(
            "WM-VLM oracle encoding requires full_vision_tokens targets; "
            f"got {target_mode!r}"
        )
    if helper_embeds.shape[0] != latent_size:
        raise ValueError(
            "Oracle image token count does not match the checkpoint: "
            f"checkpoint={latent_size}, encoded={helper_embeds.shape[0]}"
        )
    return helper_embeds.unsqueeze(0)


@torch.inference_mode()
def encode_flow_source_image_latents(
    model: Any,
    image_processor: Any,
    flow_source_image: Any,
) -> torch.Tensor:
    """Encode a hidden query-shape crop as the flow's t=0 state."""
    if bool(
        getattr(
            model.config,
            "wm_vlm_flow_source_qwen_vl_utils_preprocess",
            getattr(model.config, "wm_vlm_vl_utils_preprocess", False),
        )
    ):
        flow_source_image = preprocess_with_qwen_vl_utils(flow_source_image)
    source_batch = image_processor(
        images=[flow_source_image],
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    source_embeddings = model._encode_visual(
        source_batch["pixel_values"].to(device),
        source_batch["image_grid_thw"].to(device),
    )
    expected = (
        int(model.config.wm_vlm_latent_size),
        int(model.config.text_config.hidden_size),
    )
    if tuple(source_embeddings.shape) != expected:
        raise ValueError(
            "Query-crop flow source must preserve every post-merger vision token; "
            f"encoded={tuple(source_embeddings.shape)}, expected={expected}"
        )
    return source_embeddings.unsqueeze(0)


@torch.inference_mode()
def generate_wm_vlm_answer(
    model: Any,
    processor: Any,
    *,
    question: str,
    images: list[Any],
    max_new_tokens: int = 128,
    flow_steps: int = 8,
    flow_solver: str = "heun",
    flow_seed: int = 0,
    flow_noise_device: str = "model",
    flow_state_dtype: torch.dtype = torch.float32,
    max_latent_blocks: int | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    oracle_latents: torch.Tensor | None = None,
    flow_source_latents: torch.Tensor | None = None,
    visual_consumer_mode: str = "checkpoint",
    capture_generated_pixels: bool = False,
    capture_generated_latents: bool = False,
    generated_latent_ablation: str = "none",
    generated_latent_ablation_seed: int = 0,
    force_oracle_blocks_at_start: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> str:
    """Prepare raw inputs, call ``model.generate``, and decode new tokens."""
    if not images:
        raise ValueError("At least one problem image is required")
    messages = build_user_messages(
        question,
        len(images),
        prompt_style=getattr(model.config, "wm_vlm_prompt_style", "wm_vlm"),
    )
    if bool(
        getattr(
            model.config,
            "wm_vlm_problem_qwen_vl_utils_preprocess",
            getattr(model.config, "wm_vlm_vl_utils_preprocess", False),
        )
    ):
        images = [preprocess_with_qwen_vl_utils(image) for image in images]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    batch = processor(
        text=[rendered],
        images=images,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    prompt_length = int(batch["input_ids"].shape[1])
    answer_end_ids = tuple(
        int(token_id)
        for token_id in processor.tokenizer.encode(
            "</answer>",
            add_special_tokens=False,
        )
    )
    sequences = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p,
        flow_steps=flow_steps,
        flow_solver=flow_solver,
        flow_seed=flow_seed,
        flow_noise_device=flow_noise_device,
        flow_state_dtype=flow_state_dtype,
        max_latent_blocks=max_latent_blocks,
        oracle_latents=oracle_latents,
        flow_source_latents=flow_source_latents,
        visual_consumer_mode=visual_consumer_mode,
        capture_generated_pixels=capture_generated_pixels,
        capture_generated_latents=capture_generated_latents,
        generated_latent_ablation=generated_latent_ablation,
        generated_latent_ablation_seed=generated_latent_ablation_seed,
        force_oracle_blocks_at_start=force_oracle_blocks_at_start,
        stop_token_sequences=(answer_end_ids,) if answer_end_ids else (),
        diagnostics=diagnostics,
    )
    generated_ids = sequences[0, prompt_length:]
    return processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
