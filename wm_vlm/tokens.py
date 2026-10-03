"""Token registration and checkpoint-persisted WM-VLM configuration."""

from __future__ import annotations

from typing import Any

from .constants import (
    BRANCH_ONLY_INFERENCE_CONSUMER,
    BRANCH_ONLY_ROUTING,
    CE_CONSUMER_SOURCES,
    FLOW_SOURCE_MODES,
    FULL_VISION_TOKENS_TARGET,
    GAUSSIAN_NOISE_FLOW_SOURCE,
    GENERATED_PIXEL_CE_CONSUMER,
    LATENT_END_TOKEN,
    LATENT_PAD_TOKEN,
    LATENT_START_TOKEN,
    LAYER_PLACEMENTS,
    ORACLE_TARGET_CE_CONSUMER,
    TETRIS_LATENT_END_TOKEN,
    TETRIS_LATENT_PAD_TOKEN,
    TETRIS_LATENT_START_TOKEN,
    TETRIS_SPECIAL_TOKENS,
    TOKEN_EXPERT_QKV_ROUTING,
    TOP_LAYER_PLACEMENT,
    WM_VLM_SPECIAL_TOKENS,
    training_consumer_mode,
    variant_for,
)


def latent_token_strings(token_style: str) -> tuple[str, str, str]:
    """Return semantic (latent, start, end) token strings."""
    if token_style == "wm_vlm":
        return LATENT_PAD_TOKEN, LATENT_START_TOKEN, LATENT_END_TOKEN
    if token_style == "tetris":
        return (
            TETRIS_LATENT_PAD_TOKEN,
            TETRIS_LATENT_START_TOKEN,
            TETRIS_LATENT_END_TOKEN,
        )
    raise ValueError(
        f"Unsupported token_style={token_style!r}; expected 'wm_vlm' or 'tetris'"
    )


def _tokens_in_addition_order(token_style: str) -> tuple[str, str, str]:
    if token_style == "wm_vlm":
        return WM_VLM_SPECIAL_TOKENS
    if token_style == "tetris":
        return TETRIS_SPECIAL_TOKENS
    latent_token_strings(token_style)
    raise AssertionError("unreachable")


def add_wm_vlm_special_tokens(
    tokenizer: Any,
    *,
    token_style: str = "wm_vlm",
) -> int:
    """Add WM-VLM control tokens and verify one-token encoding."""
    addition_order = _tokens_in_addition_order(token_style)
    added = int(tokenizer.add_tokens(list(addition_order), special_tokens=True))
    for token in addition_order:
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise ValueError(
                f"WM-VLM token {token!r} must encode to exactly one token; got {encoded}"
            )
    return added


def wm_vlm_token_ids(
    tokenizer: Any,
    *,
    token_style: str = "wm_vlm",
) -> tuple[int, int, int]:
    ids = tuple(
        int(tokenizer.convert_tokens_to_ids(token))
        for token in latent_token_strings(token_style)
    )
    if len(set(ids)) != len(ids):
        raise ValueError(f"WM-VLM special tokens do not have distinct ids: {ids}")
    return ids


def configure_wm_vlm_model(
    model: Any,
    tokenizer: Any,
    *,
    wm_vlm_num_layers: int,
    latent_size: int = 8,
    flow_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
    pixel_patch_size: int | None = None,
    ce_weight: float = 0.1,
    ce_consumer_source: str = ORACLE_TARGET_CE_CONSUMER,
    flow_noise_scale: float = 1.0,
    freeze_vision: bool = True,
    token_style: str = "wm_vlm",
    layer_placement: str = TOP_LAYER_PLACEMENT,
    target_mode: str = FULL_VISION_TOKENS_TARGET,
    flow_source_mode: str = GAUSSIAN_NOISE_FLOW_SOURCE,
) -> None:
    """Configure a model before training and persist every architectural choice."""
    if latent_size < 1:
        raise ValueError(f"latent_size must be positive, got {latent_size}")
    total_layers = len(model.model.language_model.layers)
    if not 1 <= wm_vlm_num_layers <= total_layers:
        raise ValueError(
            f"wm_vlm_num_layers must be in [1, {total_layers}], "
            f"got {wm_vlm_num_layers}"
        )
    if flow_loss_weight < 0 or pixel_loss_weight < 0 or ce_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    if flow_loss_weight == 0 and pixel_loss_weight == 0 and ce_weight == 0:
        raise ValueError(
            "At least one of flow_loss_weight, pixel_loss_weight, or ce_weight "
            "must be positive"
        )
    if (
        pixel_loss_weight > 0 or ce_consumer_source == GENERATED_PIXEL_CE_CONSUMER
    ) and (pixel_patch_size is None or pixel_patch_size < 1):
        raise ValueError(
            "Pixel loss/generated-pixel CE requires a positive pixel_patch_size"
        )
    if ce_consumer_source not in CE_CONSUMER_SOURCES:
        raise ValueError(
            f"ce_consumer_source must be one of {CE_CONSUMER_SOURCES}, "
            f"got {ce_consumer_source!r}"
        )
    if flow_noise_scale <= 0:
        raise ValueError("flow_noise_scale must be positive")
    if layer_placement not in LAYER_PLACEMENTS:
        raise ValueError(
            f"layer_placement must be one of {LAYER_PLACEMENTS}, "
            f"got {layer_placement!r}"
        )
    if target_mode != FULL_VISION_TOKENS_TARGET:
        raise ValueError(
            "WM-VLM training requires the lossless full-vision-token target, "
            f"got {target_mode!r}"
        )
    if flow_source_mode not in FLOW_SOURCE_MODES:
        raise ValueError(
            f"flow_source_mode must be one of {FLOW_SOURCE_MODES}, "
            f"got {flow_source_mode!r}"
        )

    latent_id, start_id, end_id = wm_vlm_token_ids(
        tokenizer,
        token_style=token_style,
    )
    variant = variant_for(layer_placement)
    model.config.wm_vlm_variant = variant
    model.config.wm_vlm_latent_size = int(latent_size)
    model.config.wm_vlm_latent_token_id = latent_id
    model.config.wm_vlm_latent_start_id = start_id
    model.config.wm_vlm_latent_end_id = end_id
    model.config.wm_vlm_token_style = token_style
    model.config.wm_vlm_freeze_vision = bool(freeze_vision)
    model.config.wm_vlm_routing_mode = BRANCH_ONLY_ROUTING
    model.config.wm_vlm_layer_placement = layer_placement
    model.config.wm_vlm_qkv_routing = TOKEN_EXPERT_QKV_ROUTING
    model.config.text_config.wm_vlm_qkv_routing = TOKEN_EXPERT_QKV_ROUTING
    model.config.wm_vlm_ce_consumer_source = ce_consumer_source
    model.config.wm_vlm_training_consumer_mode = training_consumer_mode(
        ce_consumer_source,
    )
    model.config.wm_vlm_inference_consumer_mode = BRANCH_ONLY_INFERENCE_CONSUMER
    model.config.wm_vlm_num_layers = int(wm_vlm_num_layers)
    model.config.wm_vlm_target_mode = target_mode
    model.config.wm_vlm_flow_loss_weight = float(flow_loss_weight)
    model.config.wm_vlm_pixel_loss_weight = float(pixel_loss_weight)
    model.config.wm_vlm_pixel_reconstruction = bool(
        getattr(model.config, "wm_vlm_pixel_reconstruction", False)
        or pixel_loss_weight > 0
        or ce_consumer_source == GENERATED_PIXEL_CE_CONSUMER
    )
    if pixel_patch_size is not None:
        model.config.wm_vlm_pixel_patch_size = int(pixel_patch_size)
    model.config.wm_vlm_ce_weight = float(ce_weight)
    model.config.wm_vlm_flow_noise_scale = float(flow_noise_scale)
    model.config.wm_vlm_flow_source_mode = flow_source_mode
    model.config.wm_vlm_parallel_tokens = True
    model.config.wm_vlm_initialized = True


def validate_wm_vlm_checkpoint(model: Any, tokenizer: Any) -> None:
    token_style = getattr(model.config, "wm_vlm_token_style", "wm_vlm")
    expected = wm_vlm_token_ids(tokenizer, token_style=token_style)
    configured = (
        getattr(model.config, "wm_vlm_latent_token_id", None),
        getattr(model.config, "wm_vlm_latent_start_id", None),
        getattr(model.config, "wm_vlm_latent_end_id", None),
    )
    if configured != expected:
        names = latent_token_strings(token_style)
        detail = ", ".join(
            f"{name}: tokenizer={actual}, config={saved}"
            for name, actual, saved in zip(names, expected, configured, strict=True)
        )
        raise ValueError(f"Checkpoint/tokenizer WM-VLM token ids do not match ({detail})")
    embedding_rows = int(model.get_input_embeddings().weight.shape[0])
    if embedding_rows != len(tokenizer):
        raise ValueError(
            "Checkpoint embedding rows do not match its tokenizer: "
            f"embeddings={embedding_rows}, tokenizer={len(tokenizer)}"
        )
