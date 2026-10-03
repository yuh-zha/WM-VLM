"""Stable names and routing modes for the WM-VLM architectures."""

from __future__ import annotations

from typing import Any


LATENT_PAD_TOKEN = "<|latent_pad|>"
LATENT_START_TOKEN = "<|latent_start|>"
LATENT_END_TOKEN = "<|latent_end|>"
WM_VLM_SPECIAL_TOKENS = (
    LATENT_PAD_TOKEN,
    LATENT_START_TOKEN,
    LATENT_END_TOKEN,
)
TETRIS_LATENT_START_TOKEN = "<|lvr_start|>"
TETRIS_LATENT_PAD_TOKEN = "<|lvr_sep|>"
TETRIS_LATENT_END_TOKEN = "<|lvr_end|>"
TETRIS_SPECIAL_TOKENS = (
    TETRIS_LATENT_START_TOKEN,
    TETRIS_LATENT_PAD_TOKEN,
    TETRIS_LATENT_END_TOKEN,
)
WM_VLM_SYSTEM_PROMPT = (
    "You are a visual reasoning assistant. Think step by step to answer the question. "
    "Put textual reasoning inside <think> and </think>. Use a latent visual state when "
    "spatial imagination is needed, then put the final option letter inside <answer> "
    "and </answer>."
)
DEFAULT_REASONING_TEXT = (
    "<think>Construct an imagined top-down summary of the explored area.</think>\n"
)

WM_VLM_VARIANT = "wm_vlm"
BRANCH_ONLY_VARIANT = "wm_vlm_branch_only"
BOTTOM_BRANCH_ONLY_VARIANT = "wm_vlm_branch_only_bottom"
MIDDLE_BRANCH_ONLY_VARIANT = "wm_vlm_branch_only_middle"
LAYERS_8_11_BRANCH_ONLY_VARIANT = (
    "wm_vlm_branch_only_layers_8_11"
)
LAYERS_10_13_BRANCH_ONLY_VARIANT = (
    "wm_vlm_branch_only_layers_10_13"
)
LAYERS_14_17_BRANCH_ONLY_VARIANT = (
    "wm_vlm_branch_only_layers_14_17"
)
LAYERS_16_19_BRANCH_ONLY_VARIANT = (
    "wm_vlm_branch_only_layers_16_19"
)

BRANCH_ONLY_ROUTING = "branch_only"
TOP_LAYER_PLACEMENT = "top"
MIDDLE_LAYER_PLACEMENT = "middle"
BOTTOM_LAYER_PLACEMENT = "bottom"
LAYERS_8_11_PLACEMENT = "layers_8_11"
LAYERS_10_13_PLACEMENT = "layers_10_13"
LAYERS_14_17_PLACEMENT = "layers_14_17"
LAYERS_16_19_PLACEMENT = "layers_16_19"
FIXED_LAYER_INDICES_BY_PLACEMENT = {
    LAYERS_8_11_PLACEMENT: (8, 9, 10, 11),
    LAYERS_10_13_PLACEMENT: (10, 11, 12, 13),
    LAYERS_14_17_PLACEMENT: (14, 15, 16, 17),
    LAYERS_16_19_PLACEMENT: (16, 17, 18, 19),
}
SINGLE_LAYER_INDICES_BY_PLACEMENT = {
    f"layer_{layer_index}": (layer_index,) for layer_index in range(28)
}
LAYER_PLACEMENTS = (
    TOP_LAYER_PLACEMENT,
    MIDDLE_LAYER_PLACEMENT,
    BOTTOM_LAYER_PLACEMENT,
    *FIXED_LAYER_INDICES_BY_PLACEMENT,
    *SINGLE_LAYER_INDICES_BY_PLACEMENT,
)
TOKEN_EXPERT_QKV_ROUTING = "token_expert"
QKV_ROUTING_MODES = (TOKEN_EXPERT_QKV_ROUTING,)
FULL_VISION_TOKENS_TARGET = "full_vision_tokens"
TARGET_MODES = (FULL_VISION_TOKENS_TARGET,)
GAUSSIAN_NOISE_FLOW_SOURCE = "gaussian_noise"
QUERY_IMAGE_EMBEDDING_FLOW_SOURCE = "query_image_embedding"
FLOW_SOURCE_MODES = (
    GAUSSIAN_NOISE_FLOW_SOURCE,
    QUERY_IMAGE_EMBEDDING_FLOW_SOURCE,
)
ORACLE_TARGET_CE_CONSUMER = "oracle_target"
GENERATED_ENDPOINT_CE_CONSUMER = "generated_endpoint"
GENERATED_PIXEL_CE_CONSUMER = "generated_pixel"
CE_CONSUMER_SOURCES = (
    ORACLE_TARGET_CE_CONSUMER,
    GENERATED_ENDPOINT_CE_CONSUMER,
    GENERATED_PIXEL_CE_CONSUMER,
)
BRANCH_ONLY_TRAINING_CONSUMER = "oracle_target_full_vlm"
BRANCH_ONLY_INFERENCE_CONSUMER = "generated_endpoint_full_vlm"
BRANCH_ONLY_GENERATED_PIXEL_CONSUMER = "generated_pixel_frozen_vision_full_vlm"
BRANCH_ONLY_VARIANT_BY_PLACEMENT = {
    TOP_LAYER_PLACEMENT: BRANCH_ONLY_VARIANT,
    MIDDLE_LAYER_PLACEMENT: MIDDLE_BRANCH_ONLY_VARIANT,
    BOTTOM_LAYER_PLACEMENT: BOTTOM_BRANCH_ONLY_VARIANT,
    LAYERS_8_11_PLACEMENT: LAYERS_8_11_BRANCH_ONLY_VARIANT,
    LAYERS_10_13_PLACEMENT: LAYERS_10_13_BRANCH_ONLY_VARIANT,
    LAYERS_14_17_PLACEMENT: LAYERS_14_17_BRANCH_ONLY_VARIANT,
    LAYERS_16_19_PLACEMENT: LAYERS_16_19_BRANCH_ONLY_VARIANT,
    **{
        placement: f"wm_vlm_branch_only_{placement}"
        for placement in SINGLE_LAYER_INDICES_BY_PLACEMENT
    },
}


def training_consumer_mode(ce_consumer_source: str) -> str:
    """Describe the latent source and decoder path used by the CE pass."""
    if ce_consumer_source == ORACLE_TARGET_CE_CONSUMER:
        return BRANCH_ONLY_TRAINING_CONSUMER
    if ce_consumer_source == GENERATED_ENDPOINT_CE_CONSUMER:
        return BRANCH_ONLY_INFERENCE_CONSUMER
    if ce_consumer_source == GENERATED_PIXEL_CE_CONSUMER:
        return BRANCH_ONLY_GENERATED_PIXEL_CONSUMER
    raise ValueError(
        "ce_consumer_source must be one of "
        f"{CE_CONSUMER_SOURCES}, got {ce_consumer_source!r}"
    )


SUPPORTED_VARIANTS = frozenset(
    (
        WM_VLM_VARIANT,
        *BRANCH_ONLY_VARIANT_BY_PLACEMENT.values(),
    )
)


def variant_for(layer_placement: str) -> str:
    """Return the stable branch-only architecture name for a placement."""
    if layer_placement not in LAYER_PLACEMENTS:
        raise ValueError(
            f"layer_placement must be one of {LAYER_PLACEMENTS}, "
            f"got {layer_placement!r}"
        )
    return BRANCH_ONLY_VARIANT_BY_PLACEMENT[layer_placement]


def resolve_routing_mode(config: Any) -> str:
    """Validate that a checkpoint uses the supported branch-only routing."""
    configured = getattr(config, "wm_vlm_routing_mode", None)
    if configured is not None:
        configured = str(configured)
        if configured != BRANCH_ONLY_ROUTING:
            raise ValueError(
                "Unsupported wm_vlm_routing_mode="
                f"{configured!r}; expected {BRANCH_ONLY_ROUTING!r}"
            )
        return BRANCH_ONLY_ROUTING

    variant = getattr(config, "wm_vlm_variant", None)
    if variant in {None, WM_VLM_VARIANT, *BRANCH_ONLY_VARIANT_BY_PLACEMENT.values()}:
        return BRANCH_ONLY_ROUTING
    raise ValueError(f"Unsupported WM-VLM checkpoint variant: {variant!r}")


def resolve_layer_placement(config: Any) -> str:
    """Resolve expert placement, using top layers when unspecified."""
    configured = getattr(config, "wm_vlm_layer_placement", None)
    if configured is None:
        placement_by_variant = {
            variant: placement
            for placement, variant in BRANCH_ONLY_VARIANT_BY_PLACEMENT.items()
        }
        return placement_by_variant.get(
            getattr(config, "wm_vlm_variant", None),
            TOP_LAYER_PLACEMENT,
        )
    configured = str(configured)
    if configured not in LAYER_PLACEMENTS:
        raise ValueError(
            "Unsupported wm_vlm_layer_placement="
            f"{configured!r}; expected one of {LAYER_PLACEMENTS}"
        )
    return configured


def generation_layer_indices(
    total_layers: int,
    generation_layers: int,
    layer_placement: str,
) -> tuple[int, ...]:
    """Resolve a configured contiguous generation-layer block."""
    if not 1 <= generation_layers <= total_layers:
        raise ValueError(
            "generation_layers must be in [1, total_layers], got "
            f"{generation_layers} for total_layers={total_layers}"
        )
    if layer_placement not in LAYER_PLACEMENTS:
        raise ValueError(
            f"layer_placement must be one of {LAYER_PLACEMENTS}, "
            f"got {layer_placement!r}"
        )
    fixed_indices = (
        FIXED_LAYER_INDICES_BY_PLACEMENT
        | SINGLE_LAYER_INDICES_BY_PLACEMENT
    ).get(layer_placement)
    if fixed_indices is not None:
        if generation_layers != len(fixed_indices):
            raise ValueError(
                f"{layer_placement} requires {len(fixed_indices)} generation "
                f"layers, got {generation_layers}"
            )
        if fixed_indices[-1] >= total_layers:
            raise ValueError(
                f"{layer_placement} requires layer {fixed_indices[-1]}, but "
                f"total_layers={total_layers}"
            )
        return fixed_indices
    if layer_placement == BOTTOM_LAYER_PLACEMENT:
        start = 0
    elif layer_placement == MIDDLE_LAYER_PLACEMENT:
        start = (total_layers - generation_layers) // 2
    else:
        start = total_layers - generation_layers
    return tuple(range(start, start + generation_layers))


def resolve_qkv_routing(config: Any) -> str:
    """Resolve the per-token Q/K/V routing used by wm_vlm layers."""
    configured = getattr(config, "wm_vlm_qkv_routing", None)
    if configured is None:
        return TOKEN_EXPERT_QKV_ROUTING
    configured = str(configured)
    if configured not in QKV_ROUTING_MODES:
        raise ValueError(
            "Unsupported wm_vlm_qkv_routing="
            f"{configured!r}; expected one of {QKV_ROUTING_MODES}"
        )
    return configured


def resolve_target_mode(config: Any) -> str:
    """Resolve the lossless full-vision-token helper target."""
    configured = getattr(config, "wm_vlm_target_mode", None)
    if configured is None:
        return FULL_VISION_TOKENS_TARGET
    configured = str(configured)
    if configured not in TARGET_MODES:
        raise ValueError(
            "Unsupported wm_vlm_target_mode="
            f"{configured!r}; expected one of {TARGET_MODES}"
        )
    return configured


def resolve_flow_source_mode(config: Any) -> str:
    """Resolve the flow source, using Gaussian noise when unspecified."""
    configured = getattr(
        config,
        "wm_vlm_flow_source_mode",
        GAUSSIAN_NOISE_FLOW_SOURCE,
    )
    configured = str(configured)
    if configured not in FLOW_SOURCE_MODES:
        raise ValueError(
            "Unsupported wm_vlm_flow_source_mode="
            f"{configured!r}; expected one of {FLOW_SOURCE_MODES}"
        )
    return configured


def resolve_ce_consumer_source(config: Any) -> str:
    """Resolve the CE latent source, using oracle targets when unspecified."""
    configured = getattr(
        config,
        "wm_vlm_ce_consumer_source",
        ORACLE_TARGET_CE_CONSUMER,
    )
    configured = str(configured)
    if configured not in CE_CONSUMER_SOURCES:
        raise ValueError(
            "Unsupported wm_vlm_ce_consumer_source="
            f"{configured!r}; expected one of {CE_CONSUMER_SOURCES}"
        )
    return configured
