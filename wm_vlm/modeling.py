"""Qwen2.5-VL with routed visual-generation experts in selected layers.

The original Qwen parameters remain at their stock state-dict paths. Only the
configured contiguous ``wm_vlm_num_layers`` decoder layers gain a second,
isomorphic attention/MLP/LayerNorm branch. During flow generation, reasoning
tokens bypass VLM-only layers and use a separate final norm. Per-token attention
routing lets each token select its own Q/K/V expert, then all routed projections
participate in one global self-attention operation.

The visual branch is a conditional flow-matching velocity field. Tokens within
one image block are predicted in parallel. In a multi-block trace, each block is
predicted causally and may attend ground-truth prior image blocks during Stage-1
teacher forcing; future text and image blocks stay hidden.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLDecoderLayer,
    apply_multimodal_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import ModelOutput

from .configuration_wm_vlm import WMVLMConfig
from .generation import generate_wm_vlm_tokens
from .constants import (
    FULL_VISION_TOKENS_TARGET,
    GENERATED_ENDPOINT_CE_CONSUMER,
    GENERATED_PIXEL_CE_CONSUMER,
    BRANCH_ONLY_INFERENCE_CONSUMER,
    QUERY_IMAGE_EMBEDDING_FLOW_SOURCE,
    TOKEN_EXPERT_QKV_ROUTING,
    generation_layer_indices,
    resolve_ce_consumer_source,
    resolve_flow_source_mode,
    resolve_layer_placement,
    resolve_qkv_routing,
    resolve_routing_mode,
    resolve_target_mode,
    training_consumer_mode,
    variant_for,
)


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.LongTensor | None,
) -> torch.Tensor | None:
    if labels is None:
        return None
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )


class WMVLMBaseForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL helpers required by the WM-VLM training and decode paths."""

    def set_vision_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.visual.parameters():
            parameter.requires_grad_(trainable)

    def freeze_vision_encoder(self) -> None:
        self.set_vision_encoder_trainable(False)

    def _compute_rope_index(
        self,
        *,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor | None,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        get_rope_index = getattr(self.model, "get_rope_index", None)
        if get_rope_index is None:
            get_rope_index = getattr(super(), "get_rope_index", None)
        if get_rope_index is None:
            raise AttributeError(
                "The installed Qwen2.5-VL model does not expose get_rope_index"
            )
        return get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

    def _set_rope_deltas(self, rope_deltas: torch.Tensor | None) -> None:
        if hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = rope_deltas
        else:
            self.rope_deltas = rope_deltas

    def _get_rope_deltas(self) -> torch.Tensor | None:
        if hasattr(self.model, "rope_deltas"):
            return self.model.rope_deltas
        return getattr(self, "rope_deltas", None)

    def reset_rope_deltas(self) -> None:
        self._set_rope_deltas(None)

    def _encode_visual(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
    ) -> torch.Tensor:
        visual_inputs = pixel_values.to(dtype=self.visual.dtype)

        def encode() -> torch.Tensor:
            features = self.get_image_features(visual_inputs, image_grid_thw)
            if isinstance(features, torch.Tensor):
                return features
            return torch.cat(tuple(features), dim=0)

        if bool(getattr(self.config, "wm_vlm_freeze_vision", True)):
            with torch.no_grad():
                return encode()
        return encode()

    def _embed_user_inputs(
        self,
        input_ids: torch.LongTensor,
        *,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.LongTensor | None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is None:
            return inputs_embeds
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required when pixel_values is provided")
        image_embeds = self._encode_visual(pixel_values, image_grid_thw)
        image_mask = input_ids == self.config.image_token_id
        num_tokens = int(image_mask.sum().item())
        if num_tokens != image_embeds.shape[0]:
            raise ValueError(
                "User image features and image tokens do not match: "
                f"tokens={num_tokens}, features={image_embeds.shape[0]}"
            )
        return inputs_embeds.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(inputs_embeds),
            image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )

    def forward_generation(self, **kwargs: Any) -> Any:
        return super().forward(**kwargs)


@dataclass
class WMVLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    flow_loss: torch.FloatTensor | None = None
    pixel_loss: torch.FloatTensor | None = None
    ce_loss: torch.FloatTensor | None = None
    endpoint_mse: torch.FloatTensor | None = None
    endpoint_cosine: torch.FloatTensor | None = None
    predicted_velocity_norm: torch.FloatTensor | None = None
    target_velocity_norm: torch.FloatTensor | None = None
    target_latent_norm: torch.FloatTensor | None = None
    flow_t_mean: torch.FloatTensor | None = None
    predicted_velocity: torch.FloatTensor | None = None
    predicted_endpoint: torch.FloatTensor | None = None
    flow_source_latents: torch.FloatTensor | None = None
    target_latents: torch.FloatTensor | None = None
    predicted_pixels: torch.FloatTensor | None = None
    rope_deltas: torch.LongTensor | None = None


def _validate_reasoning_mask(
    attention_mask: torch.Tensor,
    reasoning_mask: torch.Tensor,
) -> None:
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape [batch, seq], got {tuple(attention_mask.shape)}"
        )
    if reasoning_mask.shape != attention_mask.shape:
        raise ValueError(
            "reasoning_mask must match attention_mask: "
            f"reasoning={tuple(reasoning_mask.shape)}, attention={tuple(attention_mask.shape)}"
        )
    if reasoning_mask.dtype != torch.bool:
        raise ValueError("reasoning_mask must be boolean")
    if bool((reasoning_mask & ~attention_mask.bool()).any()):
        raise ValueError("Reasoning tokens cannot occupy padded positions")


def build_parallel_reasoning_attention_masks(
    attention_mask: torch.Tensor,
    reasoning_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    sliding_window: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build causal text masks with one bidirectional reasoning-token block.

    Text remains causal.  A reasoning-token query may additionally attend every
    reasoning token in its own sample, which makes the complete visual-state
    block a simultaneous flow variable rather than an autoregressive sequence.
    """
    _validate_reasoning_mask(attention_mask, reasoning_mask)
    if not dtype.is_floating_point:
        raise ValueError(f"Attention-mask dtype must be floating point, got {dtype}")

    batch_size, sequence_length = attention_mask.shape
    device = attention_mask.device
    query_index = torch.arange(sequence_length, device=device).view(1, sequence_length, 1)
    key_index = torch.arange(sequence_length, device=device).view(1, 1, sequence_length)
    causal = key_index <= query_index
    valid_keys = attention_mask.bool().view(batch_size, 1, sequence_length)
    reasoning_pairs = reasoning_mask.view(batch_size, sequence_length, 1) & reasoning_mask.view(
        batch_size, 1, sequence_length
    )
    full_allowed = (causal | reasoning_pairs) & valid_keys

    # Avoid fully masked padded query rows, which can produce NaNs in eager and
    # SDPA attention.  These outputs are ignored by labels and downstream masks.
    invalid_queries = ~attention_mask.bool().view(batch_size, sequence_length, 1)
    diagonal = torch.eye(sequence_length, dtype=torch.bool, device=device).view(
        1, sequence_length, sequence_length
    )
    full_allowed = full_allowed | (invalid_queries & diagonal)

    mask_value = torch.finfo(dtype).min

    def additive(allowed: torch.Tensor) -> torch.Tensor:
        result = torch.full(
            (batch_size, 1, sequence_length, sequence_length),
            mask_value,
            dtype=dtype,
            device=device,
        )
        return result.masked_fill(allowed.unsqueeze(1), 0.0)

    masks = {"full_attention": additive(full_allowed)}
    if sliding_window is not None:
        if sliding_window < 1:
            raise ValueError("sliding_window must be positive when provided")
        within_window = key_index > (query_index - sliding_window)
        sliding_allowed = ((causal & within_window) | reasoning_pairs) & valid_keys
        sliding_allowed = sliding_allowed | (invalid_queries & diagonal)
        masks["sliding_attention"] = additive(sliding_allowed)
    return masks


def build_vlm_only_attention_masks(
    attention_mask: torch.Tensor,
    reasoning_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    sliding_window: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build VLM-trunk masks that completely isolate bypassed visual slots.

    Ordinary queries retain causal attention but cannot use reasoning slots as
    keys. Reasoning queries receive only a safe diagonal entry; their outputs
    are discarded and replaced by the unchanged branch input after every
    VLM-only layer.
    """
    _validate_reasoning_mask(attention_mask, reasoning_mask)
    if not dtype.is_floating_point:
        raise ValueError(f"Attention-mask dtype must be floating point, got {dtype}")

    batch_size, sequence_length = attention_mask.shape
    device = attention_mask.device
    query_index = torch.arange(sequence_length, device=device).view(1, sequence_length, 1)
    key_index = torch.arange(sequence_length, device=device).view(1, 1, sequence_length)
    causal = key_index <= query_index
    valid_nonreasoning_keys = (
        attention_mask.bool() & ~reasoning_mask
    ).view(batch_size, 1, sequence_length)
    text_queries = ~reasoning_mask.view(batch_size, sequence_length, 1)
    diagonal = torch.eye(sequence_length, dtype=torch.bool, device=device).view(
        1, sequence_length, sequence_length
    )
    reasoning_diagonal = reasoning_mask.view(batch_size, sequence_length, 1) & diagonal
    allowed = (causal & valid_nonreasoning_keys & text_queries) | reasoning_diagonal

    invalid_queries = ~attention_mask.bool().view(batch_size, sequence_length, 1)
    allowed = allowed | (invalid_queries & diagonal)
    mask_value = torch.finfo(dtype).min

    def additive(values: torch.Tensor) -> torch.Tensor:
        result = torch.full(
            (batch_size, 1, sequence_length, sequence_length),
            mask_value,
            dtype=dtype,
            device=device,
        )
        return result.masked_fill(values.unsqueeze(1), 0.0)

    masks = {"full_attention": additive(allowed)}
    if sliding_window is not None:
        if sliding_window < 1:
            raise ValueError("sliding_window must be positive when provided")
        within_window = key_index > (query_index - sliding_window)
        sliding_allowed = (
            causal & within_window & valid_nonreasoning_keys & text_queries
        ) | reasoning_diagonal
        sliding_allowed = sliding_allowed | (invalid_queries & diagonal)
        masks["sliding_attention"] = additive(sliding_allowed)
    return masks


class FlowTimeEmbedding(nn.Module):
    """Sinusoidal scalar-time embedding followed by a small MLP."""

    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        frequency_size = min(int(frequency_size), int(hidden_size))
        if frequency_size < 2:
            raise ValueError("frequency_size must be at least 2")
        if frequency_size % 2:
            frequency_size -= 1
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must have shape [batch], got {tuple(timesteps.shape)}")
        half = self.frequency_size // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phases = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((torch.cos(phases), torch.sin(phases)), dim=-1)
        parameter = next(self.mlp.parameters())
        return self.mlp(embedding.to(dtype=parameter.dtype))


class WMVLMDecoderLayer(Qwen2_5_VLDecoderLayer):
    """One stock Qwen text branch plus an isomorphic visual-state branch."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.generation_self_attn = copy.deepcopy(self.self_attn)
        self.generation_mlp = copy.deepcopy(self.mlp)
        self.generation_input_layernorm = copy.deepcopy(self.input_layernorm)
        self.generation_post_attention_layernorm = copy.deepcopy(
            self.post_attention_layernorm
        )

    def initialize_generation_from_text(self) -> None:
        self.generation_self_attn.load_state_dict(self.self_attn.state_dict())
        self.generation_mlp.load_state_dict(self.mlp.state_dict())
        self.generation_input_layernorm.load_state_dict(self.input_layernorm.state_dict())
        self.generation_post_attention_layernorm.load_state_dict(
            self.post_attention_layernorm.state_dict()
        )

    @staticmethod
    def _route(
        text_states: torch.Tensor,
        generation_states: torch.Tensor,
        reasoning_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(reasoning_mask.unsqueeze(-1), generation_states, text_states)

    def _token_routed_attention(
        self,
        *,
        text_states: torch.Tensor,
        generation_states: torch.Tensor,
        reasoning_mask: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        cache_position: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Route Q/K/V/O per token, then execute one global attention.

        A generation query attending a text key uses generation Q together
        with text K/V in the same global attention operation.
        """
        text_attention = self.self_attn
        generation_attention = self.generation_self_attn
        batch_size, sequence_length, _ = text_states.shape

        query_states = self._route(
            text_attention.q_proj(text_states),
            generation_attention.q_proj(generation_states),
            reasoning_mask,
        )
        key_states = self._route(
            text_attention.k_proj(text_states),
            generation_attention.k_proj(generation_states),
            reasoning_mask,
        )
        value_states = self._route(
            text_attention.v_proj(text_states),
            generation_attention.v_proj(generation_states),
            reasoning_mask,
        )

        query_states = query_states.view(
            batch_size,
            sequence_length,
            -1,
            text_attention.head_dim,
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size,
            sequence_length,
            -1,
            text_attention.head_dim,
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size,
            sequence_length,
            -1,
            text_attention.head_dim,
        ).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            text_attention.rope_scaling["mrope_section"],
        )

        attention_interface = eager_attention_forward
        implementation = text_attention.config._attn_implementation
        if implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[implementation]
        attention_output, attention_weights = attention_interface(
            text_attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=(
                0.0
                if not self.training
                else text_attention.attention_dropout
            ),
            scaling=text_attention.scaling,
            sliding_window=text_attention.sliding_window,
            position_ids=position_ids,
            cache_position=cache_position,
            **kwargs,
        )
        attention_output = attention_output.reshape(
            batch_size,
            sequence_length,
            -1,
        ).contiguous()
        return (
            self._route(
                text_attention.o_proj(attention_output),
                generation_attention.o_proj(attention_output),
                reasoning_mask,
            ),
            attention_weights,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        wm_vlm_token_mask: torch.Tensor | None = None,
        wm_vlm_time_embedding: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        if wm_vlm_token_mask is None or not bool(wm_vlm_token_mask.any()):
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        if wm_vlm_token_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                "wm_vlm_token_mask must match hidden states: "
                f"mask={tuple(wm_vlm_token_mask.shape)}, hidden={tuple(hidden_states.shape[:2])}"
            )
        if past_key_values is not None or use_cache:
            raise ValueError(
                "WM-VLM reasoning execution currently requires a full-sequence, "
                "cache-free forward"
            )
        if wm_vlm_time_embedding is not None and wm_vlm_time_embedding.shape != (
            hidden_states.shape[0],
            hidden_states.shape[2],
        ):
            raise ValueError(
                "wm_vlm_time_embedding must have shape [batch, hidden], got "
                f"{tuple(wm_vlm_time_embedding.shape)}"
            )

        residual = hidden_states
        text_normalized = self.input_layernorm(hidden_states)
        generation_normalized = self.generation_input_layernorm(hidden_states)
        if wm_vlm_time_embedding is not None:
            generation_normalized = generation_normalized + (
                wm_vlm_token_mask.unsqueeze(-1)
                * wm_vlm_time_embedding.unsqueeze(1).to(generation_normalized.dtype)
            )

        if resolve_qkv_routing(self.self_attn.config) != TOKEN_EXPERT_QKV_ROUTING:
            raise RuntimeError("wm_vlm requires per-token Q/K/V routing")
        if position_embeddings is None:
            raise ValueError("Token-routed attention requires position_embeddings")
        routed_attention, attention_weights = self._token_routed_attention(
            text_states=text_normalized,
            generation_states=generation_normalized,
            reasoning_mask=wm_vlm_token_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + routed_attention

        residual = hidden_states
        text_mlp = self.mlp(self.post_attention_layernorm(hidden_states))
        generation_mlp = self.generation_mlp(
            self.generation_post_attention_layernorm(hidden_states)
        )
        hidden_states = residual + self._route(text_mlp, generation_mlp, wm_vlm_token_mask)

        outputs: tuple[torch.Tensor, ...] = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
        return outputs


class WMVLMForConditionalGeneration(WMVLMBaseForConditionalGeneration):
    """Qwen2.5-VL with flow-matched, parallel WM-VLM visual tokens."""

    config_class = WMVLMConfig

    _no_split_modules = [
        "Qwen2_5_VLDecoderLayer",
        "WMVLMDecoderLayer",
        "Qwen2_5_VLVisionBlock",
    ]

    def __init__(self, config: Any) -> None:
        latent_size = int(getattr(config, "wm_vlm_latent_size", 8))
        if latent_size < 1:
            raise ValueError("wm_vlm_latent_size must be positive")
        num_wm_vlm_layers = int(getattr(config, "wm_vlm_num_layers", 4))
        total_layers = int(config.text_config.num_hidden_layers)
        if not 1 <= num_wm_vlm_layers <= total_layers:
            raise ValueError(
                f"wm_vlm_num_layers must be in [1, {total_layers}], "
                f"got {num_wm_vlm_layers}"
            )
        config.wm_vlm_latent_size = latent_size
        config.wm_vlm_num_layers = num_wm_vlm_layers
        target_mode = resolve_target_mode(config)
        config.wm_vlm_target_mode = target_mode
        config.wm_vlm_routing_mode = resolve_routing_mode(config)
        layer_placement = resolve_layer_placement(config)
        config.wm_vlm_layer_placement = layer_placement
        variant = variant_for(layer_placement)
        qkv_routing = resolve_qkv_routing(config)
        config.wm_vlm_qkv_routing = qkv_routing
        config.text_config.wm_vlm_qkv_routing = qkv_routing
        ce_consumer_source = resolve_ce_consumer_source(config)
        config.wm_vlm_ce_consumer_source = ce_consumer_source
        config.wm_vlm_flow_source_mode = resolve_flow_source_mode(config)
        config.wm_vlm_training_consumer_mode = training_consumer_mode(
            ce_consumer_source
        )
        config.wm_vlm_inference_consumer_mode = BRANCH_ONLY_INFERENCE_CONSUMER
        config.wm_vlm_variant = variant
        super().__init__(config)

        wm_vlm_layer_indices = generation_layer_indices(
            total_layers,
            num_wm_vlm_layers,
            layer_placement,
        )
        config.wm_vlm_layer_indices = list(wm_vlm_layer_indices)
        for layer_index in wm_vlm_layer_indices:
            layer = WMVLMDecoderLayer(config.text_config, layer_index)
            # The stock text keys will be populated by from_pretrained; these
            # initial values cover construction from a config and missing branch
            # keys when starting from a base Qwen checkpoint.
            layer.apply(self._init_weights)
            self.model.language_model.layers[layer_index] = layer

        hidden_size = int(config.text_config.hidden_size)
        self.generation_final_norm = copy.deepcopy(
            self.model.language_model.norm
        )
        self.flow_time_embedding = FlowTimeEmbedding(hidden_size)
        self.flow_velocity_head = nn.Linear(hidden_size, hidden_size, bias=True)
        self.flow_time_embedding.apply(self._init_weights)
        self.flow_velocity_head.apply(self._init_weights)
        if bool(
            getattr(
                config,
                "wm_vlm_pixel_reconstruction",
                False,
            )
        ):
            pixel_patch_size = int(
                getattr(
                    config,
                    "wm_vlm_pixel_patch_size",
                    int(config.vision_config.patch_size)
                    * int(config.vision_config.spatial_merge_size),
                )
            )
            if pixel_patch_size < 1:
                raise ValueError(
                    "wm_vlm_pixel_patch_size must be positive"
                )
            config.wm_vlm_pixel_patch_size = pixel_patch_size
            pixel_dimension = 3 * pixel_patch_size * pixel_patch_size
            self.pixel_reconstruction_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, pixel_dimension),
            )
            self.pixel_reconstruction_head.apply(self._init_weights)

    def save_pretrained(self, save_directory: Any, *args: Any, **kwargs: Any) -> None:
        """Save native module paths instead of Qwen's backward-compat aliases.

        Transformers rewrites Qwen-VL state-dict paths such as
        ``model.language_model`` back to an older external layout when saving.
        WM-VLM checkpoints use the current module paths so their config and
        weights describe the same architecture directly. Loading official Qwen
        initialization checkpoints still uses the inherited conversion map.
        """
        had_instance_mapping = "_checkpoint_conversion_mapping" in self.__dict__
        previous_mapping = self.__dict__.get("_checkpoint_conversion_mapping")
        self._checkpoint_conversion_mapping = {}
        try:
            return super().save_pretrained(save_directory, *args, **kwargs)
        finally:
            if had_instance_mapping:
                self._checkpoint_conversion_mapping = previous_mapping
            else:
                del self._checkpoint_conversion_mapping

    @torch.inference_mode()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: Any | None = None,
        logits_processor: Any | None = None,
        stopping_criteria: Any | None = None,
        prefix_allowed_tokens_fn: Any | None = None,
        synced_gpus: bool | None = None,
        assistant_model: Any | None = None,
        streamer: Any | None = None,
        negative_prompt_ids: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.LongTensor | GenerateDecoderOnlyOutput:
        """Run WM-VLM flow decoding behind the standard HF ``generate`` API.

        The method intentionally supports greedy and nucleus sampling with one
        sequence at a time. Beam search, assisted decoding, and streaming are
        rejected because inserting continuous visual blocks changes sequence
        length outside the stock autoregressive decoding loop.
        """

        def configured(name: str, default: Any) -> Any:
            value = kwargs.pop(name, None)
            if value is not None:
                return value
            source = generation_config or getattr(self, "generation_config", None)
            configured_value = None if source is None else getattr(source, name, None)
            return default if configured_value is None else configured_value

        input_ids = kwargs.pop("input_ids", None)
        if inputs is not None:
            if input_ids is not None:
                raise ValueError("Pass either inputs or input_ids, not both")
            input_ids = inputs
        if input_ids is None:
            raise ValueError("WM-VLM generate() requires input_ids")

        unsupported = {
            "logits_processor": logits_processor,
            "stopping_criteria": stopping_criteria,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
            "assistant_model": assistant_model,
            "streamer": streamer,
            "negative_prompt_ids": negative_prompt_ids,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
        }
        active_unsupported = [
            name
            for name, value in unsupported.items()
            if value is not None and (not hasattr(value, "__len__") or len(value) > 0)
        ]
        if synced_gpus:
            active_unsupported.append("synced_gpus")
        if active_unsupported:
            raise NotImplementedError(
                "WM-VLM custom generation does not yet support: "
                + ", ".join(active_unsupported)
            )

        num_beams = int(configured("num_beams", 1))
        num_return_sequences = int(configured("num_return_sequences", 1))
        if num_beams != 1 or num_return_sequences != 1:
            raise NotImplementedError(
                "WM-VLM generation currently supports num_beams=1 and "
                "num_return_sequences=1"
            )
        do_sample = bool(configured("do_sample", False))
        configured_temperature = configured("temperature", 1.0)
        temperature = (
            float(configured_temperature)
            if do_sample and configured_temperature is not None
            else 0.0
        )
        top_p = float(configured("top_p", 1.0))
        max_new_tokens = configured("max_new_tokens", None)
        if max_new_tokens is None:
            max_length = int(configured("max_length", input_ids.shape[1] + 128))
            max_new_tokens = max_length - int(input_ids.shape[1])
        max_new_tokens = int(max_new_tokens)

        flow_state_dtype = kwargs.pop("flow_state_dtype", torch.float32)
        if isinstance(flow_state_dtype, str):
            try:
                flow_state_dtype = {
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }[flow_state_dtype]
            except KeyError as error:
                raise ValueError(
                    "flow_state_dtype must be 'bfloat16', 'float32', or a torch dtype"
                ) from error

        return_dict_in_generate = bool(configured("return_dict_in_generate", False))
        output_scores = bool(configured("output_scores", False))
        if output_scores:
            raise NotImplementedError(
                "WM-VLM generate() does not yet retain per-step output scores"
            )

        # Accepted stock-generation parameters that do not alter this custom loop.
        for ignored_name in (
            "bos_token_id",
            "pad_token_id",
            "use_cache",
            "output_attentions",
            "output_hidden_states",
        ):
            kwargs.pop(ignored_name, None)

        generation_inputs = {
            "attention_mask": kwargs.pop("attention_mask", None),
            "pixel_values": kwargs.pop("pixel_values", None),
            "image_grid_thw": kwargs.pop("image_grid_thw", None),
            "flow_steps": int(kwargs.pop("flow_steps", 8)),
            "flow_solver": str(kwargs.pop("flow_solver", "heun")),
            "flow_seed": int(kwargs.pop("flow_seed", 0)),
            "flow_noise_device": str(kwargs.pop("flow_noise_device", "model")),
            "max_latent_blocks": kwargs.pop("max_latent_blocks", None),
            "oracle_latents": kwargs.pop("oracle_latents", None),
            "flow_source_latents": kwargs.pop("flow_source_latents", None),
            "visual_consumer_mode": str(
                kwargs.pop("visual_consumer_mode", "checkpoint")
            ),
            "capture_generated_pixels": bool(
                kwargs.pop("capture_generated_pixels", False)
            ),
            "capture_generated_latents": bool(
                kwargs.pop("capture_generated_latents", False)
            ),
            "generated_latent_ablation": str(
                kwargs.pop("generated_latent_ablation", "none")
            ),
            "generated_latent_ablation_seed": int(
                kwargs.pop("generated_latent_ablation_seed", 0)
            ),
            "force_oracle_blocks_at_start": bool(
                kwargs.pop("force_oracle_blocks_at_start", False)
            ),
            "eos_token_id": configured("eos_token_id", self.config.eos_token_id),
            "stop_token_sequences": tuple(
                kwargs.pop("stop_token_sequences", ())
            ),
            "diagnostics": kwargs.pop("diagnostics", None),
        }
        if kwargs:
            raise ValueError(
                "Unsupported WM-VLM generate() arguments: "
                + ", ".join(sorted(kwargs))
            )
        sequences = generate_wm_vlm_tokens(
            self,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            flow_state_dtype=flow_state_dtype,
            temperature=temperature,
            top_p=top_p,
            **generation_inputs,
        )
        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(sequences=sequences)
        return sequences

    @property
    def wm_vlm_layers(self) -> tuple[WMVLMDecoderLayer, ...]:
        return tuple(
            layer
            for layer in self.model.language_model.layers
            if isinstance(layer, WMVLMDecoderLayer)
        )

    def initialize_generation_branch_from_text(self) -> None:
        """Initialize a new branch from its corresponding pretrained Qwen layer."""
        for layer in self.wm_vlm_layers:
            layer.initialize_generation_from_text()
        if hasattr(self, "generation_final_norm"):
            self.generation_final_norm.load_state_dict(
                self.model.language_model.norm.state_dict()
            )

    def _generation_branch_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        parameters = [
            parameter
            for layer in self.wm_vlm_layers
            for module in (
                layer.generation_self_attn,
                layer.generation_mlp,
                layer.generation_input_layernorm,
                layer.generation_post_attention_layernorm,
            )
            for parameter in module.parameters()
        ]
        parameters.extend(self.flow_time_embedding.parameters())
        parameters.extend(self.flow_velocity_head.parameters())
        if hasattr(self, "pixel_reconstruction_head"):
            parameters.extend(self.pixel_reconstruction_head.parameters())
        if hasattr(self, "generation_final_norm"):
            parameters.extend(self.generation_final_norm.parameters())
        return tuple(parameters)

    def set_vlm_backbone_trainable(self, trainable: bool) -> None:
        """Toggle original Qwen parameters without changing generation experts."""
        generation_parameters = {
            id(parameter) for parameter in self._generation_branch_parameters()
        }
        for parameter in self.parameters():
            if id(parameter) not in generation_parameters:
                parameter.requires_grad_(trainable)
        self.set_generation_branch_trainable(True)

    def set_generation_branch_trainable(self, trainable: bool) -> None:
        """Toggle every generation-expert, time-embedding, and velocity-head parameter."""
        for parameter in self._generation_branch_parameters():
            parameter.requires_grad_(trainable)
        self.config.wm_vlm_freeze_generation_branch = not trainable

    def _require_wm_vlm_config(self) -> tuple[int, int]:
        latent_size = int(getattr(self.config, "wm_vlm_latent_size", -1))
        latent_token_id = getattr(self.config, "wm_vlm_latent_token_id", None)
        if latent_size < 1:
            raise ValueError(
                f"Checkpoint must configure a positive latent size, got {latent_size}"
            )
        if not isinstance(latent_token_id, int):
            raise ValueError("Checkpoint is missing config.wm_vlm_latent_token_id")
        return latent_size, latent_token_id

    def _reasoning_mask(self, input_ids: torch.LongTensor) -> torch.Tensor:
        latent_size, latent_token_id = self._require_wm_vlm_config()
        reasoning_mask = input_ids == latent_token_id
        counts = reasoning_mask.sum(dim=1)
        if not bool(torch.all(counts == latent_size)):
            raise ValueError(
                f"Expected exactly {latent_size} visual reasoning tokens per sample, "
                f"got {counts.tolist()}"
            )
        for row_index, row_mask in enumerate(reasoning_mask):
            positions = row_mask.nonzero(as_tuple=False).flatten()
            expected = torch.arange(
                int(positions[0]),
                int(positions[0]) + latent_size,
                device=positions.device,
            )
            if not torch.equal(positions, expected):
                raise ValueError(
                    f"Sample {row_index} reasoning tokens are not contiguous: "
                    f"{positions.tolist()}"
                )
        return reasoning_mask

    def _reasoning_block_spans(
        self,
        input_ids: torch.LongTensor,
        latent_block_counts: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, list[list[tuple[int, int]]]]:
        """Locate ordered fixed-width latent blocks in every sample."""
        latent_size, latent_token_id = self._require_wm_vlm_config()
        reasoning_mask = input_ids == latent_token_id
        if latent_block_counts is not None:
            latent_block_counts = latent_block_counts.to(
                device=input_ids.device,
                dtype=torch.long,
            )
            if tuple(latent_block_counts.shape) != (input_ids.shape[0],):
                raise ValueError(
                    "latent_block_counts must have shape "
                    f"{(input_ids.shape[0],)}, got {tuple(latent_block_counts.shape)}"
                )
        all_spans: list[list[tuple[int, int]]] = []
        for row_index, row_mask in enumerate(reasoning_mask):
            positions = row_mask.nonzero(as_tuple=False).flatten().tolist()
            spans: list[tuple[int, int]] = []
            for offset in range(0, len(positions), latent_size):
                block = positions[offset : offset + latent_size]
                if len(block) != latent_size:
                    raise ValueError(
                        f"Sample {row_index} has an incomplete latent block of "
                        f"{len(block)} tokens; expected {latent_size}"
                    )
                start = int(block[0])
                if block != list(range(start, start + latent_size)):
                    raise ValueError(
                        f"Sample {row_index} latent block is not contiguous: {block}"
                    )
                spans.append((start, start + latent_size))
            if not spans:
                raise ValueError(f"Sample {row_index} has no visual reasoning block")
            if latent_block_counts is not None:
                expected_count = int(latent_block_counts[row_index].item())
                if len(spans) != expected_count:
                    raise ValueError(
                        f"Sample {row_index} has {len(spans)} latent blocks; "
                        f"collator declared {expected_count}"
                    )
            all_spans.append(spans)
        return reasoning_mask, all_spans

    def _prepare_helper_targets(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        pixel_values_latent: torch.Tensor | None,
        image_grid_thw_latent: torch.LongTensor | None,
    ) -> torch.Tensor:
        latent_size, _ = self._require_wm_vlm_config()
        if pixel_values_latent is None or image_grid_thw_latent is None:
            raise ValueError("Flow matching requires helper-image tensors")
        if image_grid_thw_latent.shape[0] != batch_size:
            raise ValueError(
                "Expected one helper image per latent block; "
                f"blocks={batch_size}, grids={tuple(image_grid_thw_latent.shape)}"
            )

        # The helper encoder defines a fixed target.  Even when the problem
        # vision path is trainable, gradients must not move the target itself.
        with torch.no_grad():
            helper_embeddings = self._encode_visual(
                pixel_values_latent,
                image_grid_thw_latent,
            )
        merge_size = int(self.visual.spatial_merge_size)
        grid_products = image_grid_thw_latent.prod(dim=-1)
        merge_area = merge_size * merge_size
        if bool(torch.any(grid_products % merge_area != 0)):
            raise ValueError(
                "Helper-image grids are not divisible by the vision spatial "
                f"merge area {merge_area}: {image_grid_thw_latent.tolist()}"
            )
        split_sizes = (grid_products // merge_area).tolist()
        if sum(split_sizes) != helper_embeddings.shape[0]:
            raise RuntimeError(
                "Helper-image feature split does not match encoded features: "
                f"split_sizes={split_sizes}, encoded={helper_embeddings.shape[0]}"
            )
        chunks = torch.split(helper_embeddings, split_sizes)
        if resolve_target_mode(self.config) != FULL_VISION_TOKENS_TARGET:
            raise RuntimeError("wm_vlm requires full-vision-token targets")
        mismatches = [size for size in split_sizes if size != latent_size]
        if mismatches:
            raise ValueError(
                "Lossless wm_vlm targets require generation token count "
                "to equal every helper image's post-merger vision token count; "
                f"configured={latent_size}, observed={split_sizes}. No truncation "
                "or pooling is allowed."
            )
        targets = torch.stack(list(chunks), dim=0)
        return targets.to(device=device, dtype=dtype).detach()

    def reconstruct_pixels(
        self,
        visual_tokens: torch.Tensor,
        grid_hw: torch.Tensor,
    ) -> torch.Tensor:
        """Decode row-major visual endpoints into a batch of RGB images.

        Each post-merger Qwen token predicts one non-overlapping RGB patch. The
        current Tetris-2D setup uses a 19x19 token grid and 28x28 pixel patches,
        producing the exact 532x532 image seen by the helper vision processor.
        """
        if not hasattr(self, "pixel_reconstruction_head"):
            raise RuntimeError(
                "This checkpoint does not configure a pixel reconstruction head"
            )
        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must have shape [images, tokens, hidden], got "
                f"{tuple(visual_tokens.shape)}"
            )
        if grid_hw.ndim != 2 or tuple(grid_hw.shape) != (
            visual_tokens.shape[0],
            2,
        ):
            raise ValueError(
                "grid_hw must have shape [images, 2], got "
                f"{tuple(grid_hw.shape)}"
            )
        grid_hw = grid_hw.to(device=visual_tokens.device, dtype=torch.long)
        if bool((grid_hw < 1).any()):
            raise ValueError(f"Pixel grids must be positive, got {grid_hw.tolist()}")
        if not bool(torch.all(grid_hw == grid_hw[:1])):
            raise ValueError(
                "Pixel reconstruction currently requires one common image grid per "
                f"batch, got {grid_hw.tolist()}"
            )
        grid_height, grid_width = (int(value) for value in grid_hw[0].tolist())
        expected_tokens = grid_height * grid_width
        if visual_tokens.shape[1] != expected_tokens:
            raise ValueError(
                "Visual-token count does not match pixel grid: "
                f"tokens={visual_tokens.shape[1]}, grid={grid_height}x{grid_width}"
            )
        patch_size = int(self.config.wm_vlm_pixel_patch_size)
        patches = self.pixel_reconstruction_head(visual_tokens)
        expected_patch_dimension = 3 * patch_size * patch_size
        if patches.shape[-1] != expected_patch_dimension:
            raise RuntimeError(
                "Pixel head output dimension does not match configured patch size: "
                f"output={patches.shape[-1]}, expected={expected_patch_dimension}"
            )
        batch_size = visual_tokens.shape[0]
        return (
            patches.reshape(
                batch_size,
                grid_height,
                grid_width,
                3,
                patch_size,
                patch_size,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(
                batch_size,
                3,
                grid_height * patch_size,
                grid_width * patch_size,
            )
        )

    def _qwen_patchify_generated_pixels(
        self,
        pixels: torch.Tensor,
        grid_hw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        """Apply Qwen's differentiable RGB normalization and patch ordering.

        The ordinary image processor goes through PIL/NumPy and would sever the
        CE gradient.  This is the exact tensor equivalent for already-resized
        static RGB images, including temporal duplication and merge-group order.
        """
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(
                "Generated pixels must have shape [images, 3, height, width], got "
                f"{tuple(pixels.shape)}"
            )
        if grid_hw.ndim != 2 or tuple(grid_hw.shape) != (pixels.shape[0], 2):
            raise ValueError(
                "Generated-pixel grid must have shape [images, 2], got "
                f"{tuple(grid_hw.shape)}"
            )
        grid_hw = grid_hw.to(device=pixels.device, dtype=torch.long)
        if bool((grid_hw < 1).any()):
            raise ValueError(f"Generated-pixel grids must be positive: {grid_hw.tolist()}")
        if not bool(torch.all(grid_hw == grid_hw[:1])):
            raise ValueError(
                "Generated-pixel encoding currently requires a common grid: "
                f"{grid_hw.tolist()}"
            )

        vision_config = self.config.vision_config
        patch_size = int(vision_config.patch_size)
        merge_size = int(vision_config.spatial_merge_size)
        temporal_patch_size = int(vision_config.temporal_patch_size)
        configured_pixel_patch = int(self.config.wm_vlm_pixel_patch_size)
        expected_pixel_patch = patch_size * merge_size
        if configured_pixel_patch != expected_pixel_patch:
            raise ValueError(
                "Generated-pixel CE requires each endpoint token to decode exactly "
                "one Qwen post-merger patch: "
                f"configured={configured_pixel_patch}, expected={expected_pixel_patch}"
            )
        post_grid_height, post_grid_width = (
            int(value) for value in grid_hw[0].tolist()
        )
        grid_height = post_grid_height * merge_size
        grid_width = post_grid_width * merge_size
        expected_height = grid_height * patch_size
        expected_width = grid_width * patch_size
        if tuple(pixels.shape[-2:]) != (expected_height, expected_width):
            raise ValueError(
                "Generated image size does not match its Qwen grid: "
                f"pixels={tuple(pixels.shape[-2:])}, "
                f"expected={(expected_height, expected_width)}"
            )

        image_mean = getattr(
            self.config,
            "wm_vlm_pixel_image_mean",
            (0.48145466, 0.4578275, 0.40821073),
        )
        image_std = getattr(
            self.config,
            "wm_vlm_pixel_image_std",
            (0.26862954, 0.26130258, 0.27577711),
        )
        if len(image_mean) != 3 or len(image_std) != 3:
            raise ValueError("Generated-pixel image mean/std must contain three values")
        mean = pixels.new_tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = pixels.new_tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1)
        if bool((std <= 0).any()):
            raise ValueError(f"Generated-pixel image std must be positive: {image_std}")
        normalized = (pixels.float() - mean) / std
        frames = normalized.unsqueeze(1).expand(
            -1,
            temporal_patch_size,
            -1,
            -1,
            -1,
        )
        batch_size, _, channels, _, _ = frames.shape
        patches = frames.reshape(
            batch_size,
            1,
            temporal_patch_size,
            channels,
            post_grid_height,
            merge_size,
            patch_size,
            post_grid_width,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flattened = patches.reshape(
            batch_size * grid_height * grid_width,
            channels * temporal_patch_size * patch_size * patch_size,
        )
        image_grid_thw = torch.tensor(
            [1, grid_height, grid_width],
            device=pixels.device,
            dtype=torch.long,
        ).expand(batch_size, -1)
        return flattened, image_grid_thw

    def encode_generated_pixels(
        self,
        pixels: torch.Tensor,
        grid_hw: torch.Tensor,
    ) -> torch.Tensor:
        """Encode generated RGB through a weight-frozen VE with input autograd."""
        pixel_values, image_grid_thw = self._qwen_patchify_generated_pixels(
            pixels,
            grid_hw,
        )
        # Do not call the usual frozen-image helper: it intentionally uses
        # torch.no_grad(). VE parameters remain frozen via requires_grad=False,
        # while this forward retains gradients with respect to generated pixels.
        features = self.get_image_features(
            pixel_values.to(dtype=self.visual.dtype),
            image_grid_thw,
        )
        if not isinstance(features, torch.Tensor):
            features = torch.cat(tuple(features), dim=0)
        expected_tokens = int(grid_hw[0].prod().item())
        expected_shape = (
            pixels.shape[0] * expected_tokens,
            int(self.config.text_config.hidden_size),
        )
        if tuple(features.shape) != expected_shape:
            raise RuntimeError(
                "Generated-pixel VE features have unexpected shape: "
                f"encoded={tuple(features.shape)}, expected={expected_shape}"
            )
        return features.reshape(
            pixels.shape[0],
            expected_tokens,
            features.shape[-1],
        )

    def generated_pixel_consumer_tokens(
        self,
        visual_tokens: torch.Tensor,
        *,
        pixel_ablation: str = "none",
        patch_shuffle_seed: int = 0,
        return_pixels: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Decode completed endpoints to RGB and re-encode them for inference.

        Training receives the post-merger helper grid from the collator. During
        autoregressive evaluation each completed visual block has only its saved
        token count, so infer the common square grid unless a future checkpoint
        explicitly persists ``wm_vlm_pixel_grid_hw``.
        """
        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must have shape [images, tokens, hidden], got "
                f"{tuple(visual_tokens.shape)}"
            )
        configured_grid = getattr(
            self.config,
            "wm_vlm_pixel_grid_hw",
            None,
        )
        if configured_grid is None:
            side = math.isqrt(visual_tokens.shape[1])
            if side * side != visual_tokens.shape[1]:
                raise ValueError(
                    "Generated-pixel inference cannot infer a square image grid "
                    f"from {visual_tokens.shape[1]} endpoint tokens; persist "
                    "wm_vlm_pixel_grid_hw in the checkpoint"
                )
            grid_height, grid_width = side, side
        else:
            if len(configured_grid) != 2:
                raise ValueError(
                    "wm_vlm_pixel_grid_hw must contain [height, width], "
                    f"got {configured_grid!r}"
                )
            grid_height, grid_width = (int(value) for value in configured_grid)
            if grid_height * grid_width != visual_tokens.shape[1]:
                raise ValueError(
                    "Saved generated-pixel grid does not match endpoint tokens: "
                    f"grid={grid_height}x{grid_width}, "
                    f"tokens={visual_tokens.shape[1]}"
                )
        grid_hw = torch.tensor(
            [grid_height, grid_width],
            device=visual_tokens.device,
            dtype=torch.long,
        ).expand(visual_tokens.shape[0], -1)
        pixels = self.reconstruct_pixels(visual_tokens, grid_hw)
        if pixel_ablation == "none":
            consumer_pixels = pixels
        elif pixel_ablation == "white":
            consumer_pixels = torch.ones_like(pixels)
        elif pixel_ablation == "shuffle_patches":
            patch_size = int(self.config.wm_vlm_pixel_patch_size)
            batch_size, channels, _, _ = pixels.shape
            patches = (
                pixels.reshape(
                    batch_size,
                    channels,
                    grid_height,
                    patch_size,
                    grid_width,
                    patch_size,
                )
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(
                    batch_size,
                    grid_height * grid_width,
                    channels,
                    patch_size,
                    patch_size,
                )
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(patch_shuffle_seed))
            permutation = torch.randperm(
                grid_height * grid_width,
                generator=generator,
                device="cpu",
            ).to(visual_tokens.device)
            consumer_pixels = (
                patches[:, permutation]
                .reshape(
                    batch_size,
                    grid_height,
                    grid_width,
                    channels,
                    patch_size,
                    patch_size,
                )
                .permute(0, 3, 1, 4, 2, 5)
                .reshape_as(pixels)
            )
        else:
            raise ValueError(
                "pixel_ablation must be 'none', 'white', or 'shuffle_patches', "
                f"got {pixel_ablation!r}"
            )
        consumer_tokens = self.encode_generated_pixels(consumer_pixels, grid_hw)
        if return_pixels:
            return consumer_tokens, consumer_pixels
        return consumer_tokens

    def _prepare_image_flow_sources(
        self,
        *,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
        flow_source_latents: torch.Tensor | None,
        pixel_values_flow_source: torch.Tensor | None,
        image_grid_thw_flow_source: torch.LongTensor | None,
    ) -> torch.Tensor:
        """Encode one query-shape crop per sample as the flow's t=0 state."""
        latent_size, _ = self._require_wm_vlm_config()
        if flow_source_latents is not None:
            if (
                pixel_values_flow_source is not None
                or image_grid_thw_flow_source is not None
            ):
                raise ValueError(
                    "Pass precomputed flow_source_latents or flow-source image "
                    "tensors, not both"
                )
            expected = (batch_size, latent_size, hidden_size)
            if tuple(flow_source_latents.shape) != expected:
                raise ValueError(
                    f"flow_source_latents has shape {tuple(flow_source_latents.shape)}; "
                    f"expected {expected}"
                )
            return flow_source_latents.to(device=device, dtype=dtype).detach()
        if (
            pixel_values_flow_source is None
            or image_grid_thw_flow_source is None
        ):
            raise ValueError(
                "query_image_embedding flow requires flow_source_latents or both "
                "pixel_values_flow_source and image_grid_thw_flow_source"
            )
        if image_grid_thw_flow_source.shape[0] != batch_size:
            raise ValueError(
                "Expected one flow-source query crop per sample; "
                f"batch={batch_size}, grids={tuple(image_grid_thw_flow_source.shape)}"
            )

        # This experiment uses the same frozen Qwen vision encoder for the
        # unrotated query crop and the rotated helper target. The endpoints are
        # fixed observations, not trainable subgraphs of the flow objective.
        with torch.no_grad():
            source_embeddings = self._encode_visual(
                pixel_values_flow_source,
                image_grid_thw_flow_source,
            )
        merge_size = int(self.visual.spatial_merge_size)
        grid_products = image_grid_thw_flow_source.prod(dim=-1)
        merge_area = merge_size * merge_size
        if bool(torch.any(grid_products % merge_area != 0)):
            raise ValueError(
                "Flow-source image grids are not divisible by the vision spatial "
                f"merge area {merge_area}: {image_grid_thw_flow_source.tolist()}"
            )
        split_sizes = (grid_products // merge_area).tolist()
        if sum(split_sizes) != source_embeddings.shape[0]:
            raise RuntimeError(
                "Flow-source image feature split does not match encoded features: "
                f"split_sizes={split_sizes}, encoded={source_embeddings.shape[0]}"
            )
        mismatches = [size for size in split_sizes if size != latent_size]
        if mismatches:
            raise ValueError(
                "Query-crop flow sources must have exactly the same post-merger "
                "vision token count as the reasoning target; "
                f"configured={latent_size}, observed={split_sizes}. Resize/process "
                "the crop with the checkpoint's flow-source image processor."
            )
        sources = torch.stack(
            list(torch.split(source_embeddings, split_sizes)),
            dim=0,
        )
        return sources.to(device=device, dtype=dtype).detach()

    def _run_routed_model(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        reasoning_mask: torch.Tensor,
        image_grid_thw: torch.LongTensor | None,
        position_ids: torch.LongTensor | None,
        flow_timesteps: torch.Tensor | None,
        output_attentions: bool | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if position_ids is None:
            position_ids, rope_deltas = self._compute_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
            self._set_rope_deltas(rope_deltas)
        else:
            rope_deltas = self._get_rope_deltas()
        sliding_window = (
            int(self.config.text_config.sliding_window)
            if bool(getattr(self.config.text_config, "use_sliding_window", False))
            else None
        )
        masks = build_parallel_reasoning_attention_masks(
            attention_mask,
            reasoning_mask,
            dtype=inputs_embeds.dtype,
            sliding_window=sliding_window,
        )
        time_embedding = (
            None
            if flow_timesteps is None
            else self.flow_time_embedding(flow_timesteps).to(inputs_embeds.dtype)
        )
        hidden_states = self._run_branch_only_model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            reasoning_mask=reasoning_mask,
            position_ids=position_ids,
            routed_masks=masks,
            sliding_window=sliding_window,
            time_embedding=time_embedding,
            output_attentions=output_attentions,
        )
        return hidden_states, rope_deltas

    def _run_branch_only_model(
        self,
        *,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        reasoning_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        routed_masks: dict[str, torch.Tensor],
        sliding_window: int | None,
        time_embedding: torch.Tensor | None,
        output_attentions: bool | None,
    ) -> torch.Tensor:
        """Run a dual stream where visual states bypass every VLM-only layer."""
        language_model = self.model.language_model
        if position_ids.ndim == 2:
            rope_position_ids = position_ids[None, ...].expand(
                3, position_ids.shape[0], -1
            )
            text_position_ids = None
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
            text_position_ids = None
            rope_position_ids = position_ids
        else:
            raise ValueError(
                "Qwen position_ids must have shape [batch, seq], [3, batch, seq], "
                f"or [4, batch, seq]; got {tuple(position_ids.shape)}"
            )

        vlm_masks = build_vlm_only_attention_masks(
            attention_mask,
            reasoning_mask,
            dtype=inputs_embeds.dtype,
            sliding_window=sliding_window,
        )
        hidden_states = inputs_embeds
        # This is the independent generation-token stream. VLM-only layers
        # cannot transform it; generation layers update it. Keeping the latest
        # branch state (rather than always restoring inputs_embeds) is essential
        # when the generation layers are at the bottom of the model.
        branch_states = inputs_embeds
        position_embeddings = language_model.rotary_emb(
            hidden_states,
            rope_position_ids,
        )
        cache_position = torch.arange(
            inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

        for decoder_layer in language_model.layers:
            is_generation_layer = isinstance(
                decoder_layer,
                WMVLMDecoderLayer,
            )
            masks = routed_masks if is_generation_layer else vlm_masks
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=masks[decoder_layer.attention_type],
                position_ids=text_position_ids,
                past_key_values=None,
                output_attentions=bool(output_attentions),
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                wm_vlm_token_mask=reasoning_mask if is_generation_layer else None,
                wm_vlm_time_embedding=time_embedding if is_generation_layer else None,
            )
            hidden_states = layer_outputs[0]
            if is_generation_layer:
                branch_states = torch.where(
                    reasoning_mask.unsqueeze(-1),
                    hidden_states,
                    branch_states,
                )
            else:
                hidden_states = torch.where(
                    reasoning_mask.unsqueeze(-1),
                    branch_states,
                    hidden_states,
                )

        text_hidden_states = language_model.norm(hidden_states)
        generation_hidden_states = self.generation_final_norm(hidden_states)
        return torch.where(
            reasoning_mask.unsqueeze(-1),
            generation_hidden_states,
            text_hidden_states,
        )

    def _run_full_vlm_consumer(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_grid_thw: torch.LongTensor | None,
        position_ids: torch.LongTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Consume completed visual states with every original VLM layer."""
        if position_ids is None:
            position_ids, rope_deltas = self._compute_rope_index(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
            self._set_rope_deltas(rope_deltas)
        else:
            rope_deltas = self._get_rope_deltas()
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_position=torch.arange(
                input_ids.shape[1],
                device=input_ids.device,
            ),
        )
        return outputs.last_hidden_state, rope_deltas

    def _embed_with_cached_image_features(
        self,
        input_ids: torch.LongTensor,
        image_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """Embed a growing sequence without rerunning the frozen vision tower."""
        inputs_embeds = self.get_input_embeddings()(input_ids)
        image_mask = input_ids == self.config.image_token_id
        num_image_tokens = int(image_mask.sum().item())
        if image_features is None:
            if num_image_tokens:
                raise ValueError("image_features are required when input_ids contain image tokens")
            return inputs_embeds
        if image_features.ndim != 2 or image_features.shape[1] != inputs_embeds.shape[2]:
            raise ValueError(
                "image_features must have shape [num_image_tokens, hidden_size], got "
                f"{tuple(image_features.shape)}"
            )
        if num_image_tokens != image_features.shape[0]:
            raise ValueError(
                "Cached image features and image tokens do not match: "
                f"tokens={num_image_tokens}, features={image_features.shape[0]}"
            )
        return inputs_embeds.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(inputs_embeds),
            image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )

    def predict_flow_velocity(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        flow_states: torch.Tensor,
        flow_timesteps: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the parallel-token velocity field in one cache-free forward."""
        reasoning_mask = self._reasoning_mask(input_ids)
        latent_size, _ = self._require_wm_vlm_config()
        expected = (input_ids.shape[0], latent_size, self.config.text_config.hidden_size)
        if tuple(flow_states.shape) != expected:
            raise ValueError(
                f"flow_states has shape {tuple(flow_states.shape)}; expected {expected}"
            )
        if flow_timesteps.shape != (input_ids.shape[0],):
            raise ValueError(
                f"flow_timesteps must have shape {(input_ids.shape[0],)}, "
                f"got {tuple(flow_timesteps.shape)}"
            )
        if image_features is not None and pixel_values is not None:
            raise ValueError("Pass cached image_features or pixel_values, not both")
        inputs_embeds = (
            self._embed_with_cached_image_features(input_ids, image_features)
            if image_features is not None
            else self._embed_user_inputs(
                input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            reasoning_mask.unsqueeze(-1).expand_as(inputs_embeds),
            flow_states.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype).reshape(-1),
        )
        hidden_states, _ = self._run_routed_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            reasoning_mask=reasoning_mask,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            flow_timesteps=flow_timesteps,
            output_attentions=False,
        )
        reasoning_hidden = hidden_states[reasoning_mask].reshape(expected)
        return self.flow_velocity_head(reasoning_hidden)

    def conditioned_text_logits(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        visual_tokens: torch.Tensor,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Run text prediction after inserting completed visual states."""
        reasoning_mask, reasoning_spans = self._reasoning_block_spans(input_ids)
        latent_size, _ = self._require_wm_vlm_config()
        total_blocks = sum(len(spans) for spans in reasoning_spans)
        expected = (total_blocks, latent_size, self.config.text_config.hidden_size)
        if tuple(visual_tokens.shape) != expected:
            raise ValueError(
                f"visual_tokens has shape {tuple(visual_tokens.shape)}; expected {expected}"
            )
        if image_features is not None and pixel_values is not None:
            raise ValueError("Pass cached image_features or pixel_values, not both")
        inputs_embeds = (
            self._embed_with_cached_image_features(input_ids, image_features)
            if image_features is not None
            else self._embed_user_inputs(
                input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            reasoning_mask.unsqueeze(-1).expand_as(inputs_embeds),
            visual_tokens.to(inputs_embeds.dtype).reshape(-1),
        )
        hidden_states, _ = self._run_full_vlm_consumer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
        )
        return self.lm_head(hidden_states)

    def predict_next_flow_velocity(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        flow_states: torch.Tensor,
        flow_timesteps: torch.Tensor,
        prior_visual_tokens: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict one new block while consuming already generated latent blocks."""
        if input_ids.shape[0] != 1:
            raise ValueError("Sequential multi-block generation requires batch size 1")
        _, reasoning_spans = self._reasoning_block_spans(input_ids)
        spans = reasoning_spans[0]
        latent_size, _ = self._require_wm_vlm_config()
        hidden_size = int(self.config.text_config.hidden_size)
        expected_current = (1, latent_size, hidden_size)
        if tuple(flow_states.shape) != expected_current:
            raise ValueError(
                f"flow_states has shape {tuple(flow_states.shape)}; "
                f"expected {expected_current}"
            )
        if tuple(flow_timesteps.shape) != (1,):
            raise ValueError(
                f"flow_timesteps must have shape (1,), got {tuple(flow_timesteps.shape)}"
            )
        expected_prior = (len(spans) - 1, latent_size, hidden_size)
        if prior_visual_tokens is None:
            if expected_prior[0]:
                raise ValueError(
                    f"Expected {expected_prior[0]} prior visual blocks, got none"
                )
        elif tuple(prior_visual_tokens.shape) != expected_prior:
            raise ValueError(
                "prior_visual_tokens has shape "
                f"{tuple(prior_visual_tokens.shape)}; expected {expected_prior}"
            )
        if image_features is not None and pixel_values is not None:
            raise ValueError("Pass cached image_features or pixel_values, not both")
        inputs_embeds = (
            self._embed_with_cached_image_features(input_ids, image_features)
            if image_features is not None
            else self._embed_user_inputs(
                input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        )
        for block_index, (start, end) in enumerate(spans[:-1]):
            assert prior_visual_tokens is not None
            inputs_embeds[:, start:end, :] = prior_visual_tokens[
                block_index
            ].to(inputs_embeds.dtype).unsqueeze(0)
        current_start, current_end = spans[-1]
        inputs_embeds[:, current_start:current_end, :] = flow_states.to(
            inputs_embeds.dtype
        )
        current_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        current_mask[:, current_start:current_end] = True
        hidden_states, _ = self._run_routed_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            reasoning_mask=current_mask,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            flow_timesteps=flow_timesteps,
            output_attentions=False,
        )
        return self.flow_velocity_head(hidden_states[current_mask]).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        pixel_values_latent: torch.Tensor | None = None,
        image_grid_thw_latent: torch.LongTensor | None = None,
        pixel_values_flow_source: torch.Tensor | None = None,
        image_grid_thw_flow_source: torch.LongTensor | None = None,
        flow_source_latents: torch.Tensor | None = None,
        flow_timesteps: torch.Tensor | None = None,
        flow_noise: torch.Tensor | None = None,
        pixel_reconstruction_targets: torch.Tensor | None = None,
        pixel_reconstruction_grid_hw: torch.LongTensor | None = None,
        consumer_latent_override: torch.Tensor | None = None,
        latent_block_counts: torch.LongTensor | None = None,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> WMVLMOutput | tuple:
        del (
            use_cache,
            output_hidden_states,
            rope_deltas,
            cache_position,
            second_per_grid_ts,
            num_items_in_batch,
        )
        if input_ids is None or attention_mask is None:
            raise ValueError("Training requires input_ids and attention_mask")
        if inputs_embeds is not None:
            raise ValueError("Pass token ids; flow states are constructed internally")
        if past_key_values is not None:
            raise ValueError("Training does not accept external past_key_values")
        if pixel_values_videos is not None or video_grid_thw is not None:
            raise ValueError("wm_vlm currently supports images, not video")
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        reasoning_mask, reasoning_spans = self._reasoning_block_spans(
            input_ids,
            latent_block_counts,
        )
        base_inputs = self._embed_user_inputs(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        batch_size, _, hidden_size = base_inputs.shape
        block_counts = [len(spans) for spans in reasoning_spans]
        total_blocks = sum(block_counts)
        if any(count > 1 for count in block_counts) and batch_size != 1:
            raise ValueError(
                "Multi-block teacher-forcing currently requires per-device batch size 1"
            )
        targets = self._prepare_helper_targets(
            batch_size=total_blocks,
            device=base_inputs.device,
            dtype=base_inputs.dtype,
            pixel_values_latent=pixel_values_latent,
            image_grid_thw_latent=image_grid_thw_latent,
        )

        if flow_timesteps is None:
            timestep_epsilon = torch.finfo(torch.float32).eps
            flow_timesteps = torch.rand(
                total_blocks,
                device=base_inputs.device,
                dtype=torch.float32,
            )
            flow_timesteps = flow_timesteps.mul(1.0 - 2.0 * timestep_epsilon).add(
                timestep_epsilon
            )
        else:
            flow_timesteps = flow_timesteps.to(device=base_inputs.device, dtype=torch.float32)
        if flow_timesteps.shape != (total_blocks,):
            raise ValueError(
                f"flow_timesteps must have shape {(total_blocks,)}, got {tuple(flow_timesteps.shape)}"
            )
        if bool(((flow_timesteps < 0) | (flow_timesteps > 1)).any()):
            raise ValueError("flow_timesteps must lie in [0, 1]")

        flow_source_mode = resolve_flow_source_mode(self.config)
        if flow_source_mode == QUERY_IMAGE_EMBEDDING_FLOW_SOURCE:
            if flow_noise is not None:
                raise ValueError(
                    "query_image_embedding flow cannot also receive Gaussian flow_noise"
                )
            flow_source = self._prepare_image_flow_sources(
                batch_size=total_blocks,
                hidden_size=hidden_size,
                device=base_inputs.device,
                dtype=base_inputs.dtype,
                flow_source_latents=flow_source_latents,
                pixel_values_flow_source=pixel_values_flow_source,
                image_grid_thw_flow_source=image_grid_thw_flow_source,
            )
        else:
            if any(
                value is not None
                for value in (
                    pixel_values_flow_source,
                    image_grid_thw_flow_source,
                    flow_source_latents,
                )
            ):
                raise ValueError(
                    "Gaussian flow checkpoints cannot receive query-image flow sources"
                )
            noise_scale = float(
                getattr(self.config, "wm_vlm_flow_noise_scale", 1.0)
            )
            if flow_noise is None:
                flow_source = torch.randn_like(targets) * noise_scale
            else:
                if flow_noise.shape != targets.shape:
                    raise ValueError(
                        f"flow_noise has shape {tuple(flow_noise.shape)}; "
                        f"expected {tuple(targets.shape)}"
                    )
                flow_source = flow_noise.to(device=targets.device, dtype=targets.dtype)
        interpolation = flow_timesteps.to(targets.dtype).view(total_blocks, 1, 1)
        flow_states = (1.0 - interpolation) * flow_source + interpolation * targets
        target_velocity = targets - flow_source

        # A multi-block trace is trained causally, one generated block at a time.
        # For block k, all earlier blocks are inserted as clean ground-truth vision
        # embeddings and the sequence is cropped before every future text/image.
        # This is explicit Stage-1 teacher forcing across latent blocks.
        if total_blocks == batch_size:
            flow_length = max(spans[0][1] for spans in reasoning_spans)
            flow_input_ids = input_ids[:, :flow_length]
            flow_attention_mask = attention_mask[:, :flow_length]
            flow_reasoning_mask = reasoning_mask[:, :flow_length]
            flow_inputs = base_inputs[:, :flow_length, :].masked_scatter(
                flow_reasoning_mask.unsqueeze(-1).expand_as(
                    base_inputs[:, :flow_length, :]
                ),
                flow_states.reshape(-1),
            )
            flow_hidden, rope_deltas = self._run_routed_model(
                input_ids=flow_input_ids,
                attention_mask=flow_attention_mask,
                inputs_embeds=flow_inputs,
                reasoning_mask=flow_reasoning_mask,
                image_grid_thw=image_grid_thw,
                position_ids=(
                    None if position_ids is None else position_ids[..., :flow_length]
                ),
                flow_timesteps=flow_timesteps,
                output_attentions=output_attentions,
            )
            predicted_velocity = self.flow_velocity_head(
                flow_hidden[flow_reasoning_mask].reshape_as(targets)
            )
        else:
            spans = reasoning_spans[0]
            predicted_blocks: list[torch.Tensor] = []
            rope_deltas = None
            for block_index, (start, end) in enumerate(spans):
                block_input_ids = input_ids[:, :end]
                block_attention_mask = attention_mask[:, :end]
                current_mask = torch.zeros_like(block_input_ids, dtype=torch.bool)
                current_mask[:, start:end] = True
                block_inputs = base_inputs[:, :end, :].clone()
                for prior_index, (prior_start, prior_end) in enumerate(
                    spans[:block_index]
                ):
                    block_inputs[:, prior_start:prior_end, :] = targets[
                        prior_index
                    ].unsqueeze(0)
                block_inputs[:, start:end, :] = flow_states[block_index].unsqueeze(0)
                block_hidden, rope_deltas = self._run_routed_model(
                    input_ids=block_input_ids,
                    attention_mask=block_attention_mask,
                    inputs_embeds=block_inputs,
                    reasoning_mask=current_mask,
                    image_grid_thw=image_grid_thw,
                    position_ids=(
                        None if position_ids is None else position_ids[..., :end]
                    ),
                    flow_timesteps=flow_timesteps[block_index : block_index + 1],
                    output_attentions=output_attentions,
                )
                predicted_blocks.append(
                    self.flow_velocity_head(block_hidden[current_mask])
                )
            predicted_velocity = torch.stack(predicted_blocks, dim=0)

        predicted_float = predicted_velocity.float()
        velocity_float = target_velocity.float()
        flow_loss = F.mse_loss(predicted_float, velocity_float)
        predicted_endpoint = flow_states + (1.0 - interpolation) * predicted_velocity
        endpoint_float = predicted_endpoint.float()
        targets_float = targets.float()
        endpoint_mse = F.mse_loss(endpoint_float, targets_float)
        endpoint_cosine = F.cosine_similarity(
            endpoint_float,
            targets_float,
            dim=-1,
        ).mean()

        pixel_weight = float(
            getattr(self.config, "wm_vlm_pixel_loss_weight", 0.0)
        )
        ce_weight = float(getattr(self.config, "wm_vlm_ce_weight", 0.1))
        ce_consumer_source = resolve_ce_consumer_source(self.config)
        needs_predicted_pixels = pixel_weight > 0 or (
            ce_weight > 0 and ce_consumer_source == GENERATED_PIXEL_CE_CONSUMER
        )
        pixel_loss: torch.Tensor | None = None
        predicted_pixels: torch.Tensor | None = None
        if needs_predicted_pixels:
            if pixel_reconstruction_grid_hw is None:
                raise ValueError(
                    "Pixel reconstruction requires pixel_reconstruction_grid_hw"
                )
            predicted_pixels = self.reconstruct_pixels(
                predicted_endpoint,
                pixel_reconstruction_grid_hw,
            )
        if pixel_weight > 0:
            if pixel_reconstruction_targets is None:
                raise ValueError(
                    "Positive pixel loss requires pixel_reconstruction_targets"
                )
            assert predicted_pixels is not None
            if tuple(predicted_pixels.shape) != tuple(
                pixel_reconstruction_targets.shape
            ):
                raise ValueError(
                    "Predicted and target pixel shapes differ: "
                    f"predicted={tuple(predicted_pixels.shape)}, "
                    f"target={tuple(pixel_reconstruction_targets.shape)}"
                )
            pixel_loss = F.mse_loss(
                predicted_pixels.float(),
                pixel_reconstruction_targets.to(
                    device=predicted_pixels.device,
                    dtype=torch.float32,
                ),
            )

        ce_loss: torch.Tensor | None = None
        logits: torch.Tensor | None = None
        if ce_weight > 0:
            if labels is None:
                raise ValueError("labels are required when wm_vlm_ce_weight > 0")
            # The generated-endpoint option matches inference and keeps the CE
            # graph attached so answer supervision also reaches the flow branch.
            if consumer_latent_override is not None:
                if tuple(consumer_latent_override.shape) != tuple(targets.shape):
                    raise ValueError(
                        "consumer_latent_override has shape "
                        f"{tuple(consumer_latent_override.shape)}; expected "
                        f"{tuple(targets.shape)}"
                    )
                # Evaluation-only diagnostic hook: replace the continuous block
                # seen by the language consumer without changing the flow pass.
                consumer_tokens = consumer_latent_override.to(
                    device=targets.device,
                    dtype=targets.dtype,
                )
            else:
                if ce_consumer_source == GENERATED_PIXEL_CE_CONSUMER:
                    assert predicted_pixels is not None
                    assert pixel_reconstruction_grid_hw is not None
                    consumer_pixels = predicted_pixels
                    if bool(
                        getattr(
                            self.config,
                            "wm_vlm_freeze_generation_branch",
                            False,
                        )
                    ):
                        consumer_pixels = consumer_pixels.detach()
                    consumer_tokens = self.encode_generated_pixels(
                        consumer_pixels,
                        pixel_reconstruction_grid_hw,
                    )
                elif ce_consumer_source == GENERATED_ENDPOINT_CE_CONSUMER:
                    consumer_tokens = predicted_endpoint
                    if bool(
                        getattr(
                            self.config,
                            "wm_vlm_freeze_generation_branch",
                            False,
                        )
                    ):
                        # A frozen producer is a stop-gradient control: CE updates
                        # only the language consumer, including the VLM weights.
                        consumer_tokens = consumer_tokens.detach()
                else:
                    consumer_tokens = targets
            consumer_inputs = base_inputs.masked_scatter(
                reasoning_mask.unsqueeze(-1).expand_as(base_inputs),
                consumer_tokens.reshape(-1),
            )
            clean_hidden, clean_rope_deltas = self._run_full_vlm_consumer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=consumer_inputs,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
            )
            logits = self.lm_head(clean_hidden)
            ce_loss = causal_lm_loss(logits, labels)
            if ce_loss is None:
                raise RuntimeError("Failed to compute text cross-entropy")
            rope_deltas = clean_rope_deltas

        flow_weight = float(
            getattr(self.config, "wm_vlm_flow_loss_weight", 1.0)
        )
        loss = flow_weight * flow_loss if flow_weight > 0 else None
        if pixel_loss is not None:
            weighted_pixel = pixel_weight * pixel_loss
            loss = weighted_pixel if loss is None else loss + weighted_pixel
        if ce_loss is not None:
            weighted_ce = ce_weight * ce_loss
            loss = weighted_ce if loss is None else loss + weighted_ce
        if loss is None:
            raise RuntimeError("At least one positive training loss weight is required")
        result = WMVLMOutput(
            loss=loss,
            logits=logits,
            flow_loss=flow_loss,
            pixel_loss=pixel_loss,
            ce_loss=ce_loss,
            endpoint_mse=endpoint_mse,
            endpoint_cosine=endpoint_cosine,
            predicted_velocity_norm=predicted_float.norm(dim=-1).mean(),
            target_velocity_norm=velocity_float.norm(dim=-1).mean(),
            target_latent_norm=targets_float.norm(dim=-1).mean(),
            flow_t_mean=flow_timesteps.float().mean(),
            predicted_velocity=predicted_velocity,
            predicted_endpoint=predicted_endpoint,
            flow_source_latents=flow_source,
            target_latents=targets,
            predicted_pixels=predicted_pixels,
            rope_deltas=rope_deltas,
        )
        return result if return_dict else result.to_tuple()
    GENERATED_ENDPOINT_CE_CONSUMER,
