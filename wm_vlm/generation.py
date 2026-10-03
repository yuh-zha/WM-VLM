"""ODE sampling and sequential multi-block decoding for wm_vlm."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .constants import (
    GAUSSIAN_NOISE_FLOW_SOURCE,
    GENERATED_PIXEL_CE_CONSUMER,
    QUERY_IMAGE_EMBEDDING_FLOW_SOURCE,
    resolve_ce_consumer_source,
    resolve_flow_source_mode,
)


def _sample_token(logits: torch.Tensor, *, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(logits.argmax(dim=-1).item())
    scores = logits.float() / temperature
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        probabilities = torch.softmax(sorted_scores, dim=-1)
        remove = torch.cumsum(probabilities, dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_scores,
        )
    return int(
        torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1).item()
    )


def _eos_ids(ids: Any) -> set[int]:
    if hasattr(ids, "eos_token_id"):
        ids = ids.eos_token_id
    if ids is None:
        return set()
    if isinstance(ids, int):
        return {ids}
    return {int(token_id) for token_id in ids}


def _ends_with_any(
    token_ids: list[int],
    stop_token_sequences: tuple[tuple[int, ...], ...],
) -> bool:
    return any(
        len(sequence) <= len(token_ids)
        and tuple(token_ids[-len(sequence) :]) == sequence
        for sequence in stop_token_sequences
        if sequence
    )


def _validate_optional_latent_blocks(
    model: Any,
    latents: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Accept one or more fixed-width latent blocks for sequential decoding."""
    if latents.ndim != 3 or latents.shape[0] < 1:
        raise ValueError(
            f"{name} must have shape [num_blocks, latent_size, hidden_size], got "
            f"{tuple(latents.shape)}"
        )
    expected_size = int(model.config.wm_vlm_latent_size)
    expected_hidden = int(model.config.text_config.hidden_size)
    if tuple(latents.shape[1:]) != (expected_size, expected_hidden):
        raise ValueError(
            f"{name} has trailing shape {tuple(latents.shape[1:])}; expected "
            f"{(expected_size, expected_hidden)}"
        )
    parameter = next(model.parameters())
    return latents.to(device=parameter.device, dtype=parameter.dtype)


def integrate_flow_tokens(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_states: torch.Tensor,
    *,
    num_steps: int,
    solver: str = "heun",
    state_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, int]:
    """Integrate from t=0 to t=1 while updating every visual token together."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError(f"Unsupported flow solver {solver!r}; expected 'euler' or 'heun'")
    if state_dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError(
            "Flow state dtype must be torch.bfloat16 or torch.float32; "
            f"got {state_dtype}"
        )
    states = initial_states.to(dtype=state_dtype)
    batch_size = states.shape[0]
    step_size = 1.0 / num_steps
    function_evaluations = 0
    for step in range(num_steps):
        time = torch.full(
            (batch_size,),
            step / num_steps,
            dtype=torch.float32,
            device=states.device,
        )
        velocity = velocity_fn(states, time).to(dtype=state_dtype)
        function_evaluations += 1
        if velocity.shape != states.shape:
            raise ValueError(
                f"velocity_fn returned {tuple(velocity.shape)} for states {tuple(states.shape)}"
            )
        if solver == "euler":
            states = states + step_size * velocity
            continue
        proposal = states + step_size * velocity
        next_time = torch.full_like(time, (step + 1) / num_steps)
        next_velocity = velocity_fn(proposal, next_time).to(dtype=state_dtype)
        function_evaluations += 1
        states = states + 0.5 * step_size * (velocity + next_velocity)
    return states, function_evaluations


def _cached_problem_features(model: Any, batch: dict[str, Any]) -> torch.Tensor | None:
    pixel_values = batch.get("pixel_values")
    if pixel_values is None:
        return None
    image_grid_thw = batch.get("image_grid_thw")
    if image_grid_thw is None:
        raise ValueError("image_grid_thw is required with pixel_values")
    return model._encode_visual(pixel_values, image_grid_thw)


def _stock_text_logits(
    model: Any,
    *,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.LongTensor | None,
    image_features: torch.Tensor | None,
) -> torch.Tensor:
    inputs_embeds = model._embed_with_cached_image_features(input_ids, image_features)
    position_ids, _ = model._compute_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    outputs = model.forward_generation(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
        cache_position=torch.arange(input_ids.shape[1], device=input_ids.device),
    )
    return outputs.logits


def ablate_generated_latent_tokens(
    tokens: torch.Tensor,
    *,
    mode: str,
    seed: int,
) -> torch.Tensor:
    """Apply a deterministic post-flow ablation before the VLM consumes tokens."""
    if tokens.ndim != 3:
        raise ValueError(
            "Generated latent tokens must have shape [batch,tokens,hidden], got "
            f"{tuple(tokens.shape)}"
        )
    if mode == "none":
        return tokens
    if mode == "zero":
        return torch.zeros_like(tokens)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if mode == "shuffle_tokens":
        permutation = torch.randperm(
            tokens.shape[-2],
            generator=generator,
            device="cpu",
        ).to(tokens.device)
        return tokens.index_select(-2, permutation)
    if mode == "random_matched_stats":
        source = tokens.detach().float()
        reduction_dims = tuple(range(1, source.ndim))
        source_mean = source.mean(dim=reduction_dims, keepdim=True)
        source_std = source.std(
            dim=reduction_dims,
            keepdim=True,
            unbiased=False,
        )
        noise = torch.randn(
            tokens.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        noise_mean = noise.mean(dim=reduction_dims, keepdim=True)
        noise_std = noise.std(
            dim=reduction_dims,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1e-12)
        matched = (noise - noise_mean) / noise_std
        matched = matched.to(source.device) * source_std + source_mean
        return matched.to(dtype=tokens.dtype)
    raise ValueError(
        "mode must be one of none, zero, random_matched_stats, shuffle_tokens; "
        f"got {mode!r}"
    )


@torch.inference_mode()
def generate_wm_vlm_tokens(
    model: Any,
    *,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
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
    eos_token_id: int | list[int] | tuple[int, ...] | None = None,
    stop_token_sequences: tuple[tuple[int, ...], ...] = (),
    diagnostics: dict[str, Any] | None = None,
) -> torch.LongTensor:
    """Generate token ids while sampling fixed-size parallel visual blocks."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            "WM-VLM generation currently requires input_ids with batch size 1; "
            f"got {tuple(input_ids.shape)}"
        )
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            "attention_mask must match input_ids; "
            f"got {tuple(attention_mask.shape)} and {tuple(input_ids.shape)}"
        )
    if pixel_values is not None and image_grid_thw is None:
        raise ValueError("image_grid_thw is required when pixel_values are provided")
    if capture_generated_latents and diagnostics is None:
        raise ValueError("Latent capture requires a diagnostics dictionary")
    supported_latent_ablations = {
        "none",
        "zero",
        "random_matched_stats",
        "shuffle_tokens",
    }
    if generated_latent_ablation not in supported_latent_ablations:
        raise ValueError(
            "generated_latent_ablation must be one of "
            f"{sorted(supported_latent_ablations)}, got "
            f"{generated_latent_ablation!r}"
        )
    if force_oracle_blocks_at_start and oracle_latents is None:
        raise ValueError("Forced oracle blocks require oracle_latents")
    if generated_latent_ablation != "none" and oracle_latents is not None:
        raise ValueError("Generated-latent ablation cannot be combined with oracle latents")
    if flow_noise_device not in {"cpu", "model"}:
        raise ValueError(
            "flow_noise_device must be 'cpu' or 'model'; "
            f"got {flow_noise_device!r}"
        )
    if flow_state_dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError(
            "flow_state_dtype must be torch.bfloat16 or torch.float32; "
            f"got {flow_state_dtype}"
        )
    if oracle_latents is not None:
        oracle_latents = _validate_optional_latent_blocks(
            model,
            oracle_latents,
            name="oracle_latents",
        )
    flow_source_mode = resolve_flow_source_mode(model.config)
    if oracle_latents is not None and flow_source_latents is not None:
        raise ValueError("Pass oracle_latents or flow_source_latents, not both")
    if flow_source_latents is not None:
        flow_source_latents = _validate_optional_latent_blocks(
            model,
            flow_source_latents,
            name="flow_source_latents",
        )
    if (
        oracle_latents is None
        and flow_source_mode == QUERY_IMAGE_EMBEDDING_FLOW_SOURCE
        and flow_source_latents is None
    ):
        raise ValueError(
            "This checkpoint requires query-crop vision embeddings as its flow source"
        )
    if (
        flow_source_mode == GAUSSIAN_NOISE_FLOW_SOURCE
        and flow_source_latents is not None
    ):
        raise ValueError(
            "Gaussian-flow checkpoints cannot receive image-embedding flow sources"
        )

    device = next(model.parameters()).device
    if input_ids.device != device or attention_mask.device != device:
        raise ValueError(
            "Move input_ids and attention_mask to the model device before generate(): "
            f"model={device}, input_ids={input_ids.device}, "
            f"attention_mask={attention_mask.device}"
        )
    if pixel_values is not None and pixel_values.device != device:
        raise ValueError(
            "Move pixel_values to the model device before generate(): "
            f"model={device}, pixel_values={pixel_values.device}"
        )
    if image_grid_thw is not None and image_grid_thw.device != device:
        raise ValueError(
            "Move image_grid_thw to the model device before generate(): "
            f"model={device}, image_grid_thw={image_grid_thw.device}"
        )
    image_features = (
        None
        if pixel_values is None
        else model._encode_visual(pixel_values, image_grid_thw)
    )
    model.reset_rope_deltas()

    latent_id = int(model.config.wm_vlm_latent_token_id)
    latent_start_id = int(model.config.wm_vlm_latent_start_id)
    latent_end_id = int(model.config.wm_vlm_latent_end_id)
    latent_size = int(model.config.wm_vlm_latent_size)
    if latent_size < 1:
        raise ValueError(f"Expected a positive visual-token count, got {latent_size}")
    if max_latent_blocks is None:
        max_latent_blocks = int(
            getattr(model.config, "wm_vlm_max_latent_blocks", 1)
        )
    if max_latent_blocks < 1:
        raise ValueError("max_latent_blocks must be positive")
    eos_ids = _eos_ids(eos_token_id)
    generated_ids: list[int] = []
    visual_blocks: list[torch.Tensor] = []
    consumer_visual_blocks: list[torch.Tensor] = []
    generated_pixel_images: list[torch.Tensor] = []
    ce_consumer_source = resolve_ce_consumer_source(model.config)
    supported_consumer_modes = {
        "checkpoint",
        "generated_pixel",
        "white_pixel",
        "shuffle_pixel_patches",
        "generated_endpoint",
    }
    if visual_consumer_mode not in supported_consumer_modes:
        raise ValueError(
            f"visual_consumer_mode must be one of {sorted(supported_consumer_modes)}, "
            f"got {visual_consumer_mode!r}"
        )
    if oracle_latents is not None:
        active_visual_consumer = "oracle_target"
    elif visual_consumer_mode == "checkpoint":
        active_visual_consumer = (
            "generated_pixel"
            if ce_consumer_source == GENERATED_PIXEL_CE_CONSUMER
            else "generated_endpoint"
        )
    else:
        active_visual_consumer = visual_consumer_mode
    use_generated_pixel_consumer = active_visual_consumer in {
        "generated_pixel",
        "white_pixel",
        "shuffle_pixel_patches",
    }
    pixel_ablation = {
        "generated_pixel": "none",
        "white_pixel": "white",
        "shuffle_pixel_patches": "shuffle_patches",
    }.get(active_visual_consumer)
    flow_nfe = 0
    latent_blocks = 0

    # A Stage-1 checkpoint has never learned the CE-side sentinel policy, so it
    # generally cannot emit <|latent_start|> by itself.  This diagnostic mode
    # inserts the supplied oracle blocks as an assistant prefix, then lets the
    # unchanged model generate the answer conditioned on those exact features.
    if force_oracle_blocks_at_start:
        assert oracle_latents is not None
        forced_blocks = min(max_latent_blocks, int(oracle_latents.shape[0]))
        for block_index in range(forced_blocks):
            start_id = torch.tensor(
                [[latent_start_id]], dtype=input_ids.dtype, device=device
            )
            input_ids = torch.cat((input_ids, start_id), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
                ),
                dim=1,
            )
            generated_ids.append(latent_start_id)
            latent_ids = torch.full(
                (1, latent_size),
                latent_id,
                dtype=input_ids.dtype,
                device=device,
            )
            input_ids = torch.cat((input_ids, latent_ids), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (1, latent_size),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ),
                dim=1,
            )
            generated_ids.extend([latent_id] * latent_size)
            current_visual_tokens = oracle_latents[block_index : block_index + 1]
            visual_blocks.append(current_visual_tokens)
            consumer_visual_blocks.append(current_visual_tokens)
            end_id = torch.tensor([[latent_end_id]], dtype=input_ids.dtype, device=device)
            input_ids = torch.cat((input_ids, end_id), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
                ),
                dim=1,
            )
            generated_ids.append(latent_end_id)
            latent_blocks += 1

    for _ in range(max_new_tokens):
        if not visual_blocks:
            logits = _stock_text_logits(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
                image_features=image_features,
            )
        else:
            logits = model.conditioned_text_logits(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_tokens=torch.cat(consumer_visual_blocks, dim=0),
                image_grid_thw=image_grid_thw,
                image_features=image_features,
            )
        emitted_id = _sample_token(
            logits[:, -1, :],
            temperature=temperature,
            top_p=top_p,
        )
        generated_ids.append(emitted_id)
        new_id = torch.tensor([[emitted_id]], dtype=input_ids.dtype, device=device)
        input_ids = torch.cat((input_ids, new_id), dim=1)
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
            ),
            dim=1,
        )

        if emitted_id == latent_start_id and latent_blocks >= max_latent_blocks:
            break

        if emitted_id == latent_start_id:
            latent_blocks += 1
            latent_ids = torch.full(
                (1, latent_size),
                latent_id,
                dtype=input_ids.dtype,
                device=device,
            )
            input_ids = torch.cat((input_ids, latent_ids), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (1, latent_size),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ),
                dim=1,
            )
            generated_ids.extend([latent_id] * latent_size)
            block_index = latent_blocks - 1
            if oracle_latents is not None:
                if block_index >= oracle_latents.shape[0]:
                    raise ValueError(
                        f"Decoder requested latent block {latent_blocks}, but only "
                        f"{oracle_latents.shape[0]} oracle blocks were supplied"
                    )
                current_visual_tokens = oracle_latents[
                    block_index : block_index + 1
                ]
            else:
                if flow_source_latents is not None:
                    source_index = min(block_index, flow_source_latents.shape[0] - 1)
                    initial = flow_source_latents[
                        source_index : source_index + 1
                    ].to(
                        device=device,
                        dtype=flow_state_dtype,
                    )
                else:
                    hidden_size = int(model.config.text_config.hidden_size)
                    sampling_device = (
                        torch.device("cpu")
                        if flow_noise_device == "cpu"
                        else device
                    )
                    generator = torch.Generator(device=sampling_device)
                    generator.manual_seed(flow_seed)
                    initial = torch.randn(
                        (1, latent_size, hidden_size),
                        generator=generator,
                        device=sampling_device,
                        dtype=torch.float32,
                    ) * float(model.config.wm_vlm_flow_noise_scale)
                    initial = initial.to(device=device, dtype=flow_state_dtype)

                def velocity(states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
                    return model.predict_next_flow_velocity(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        flow_states=states,
                        flow_timesteps=times,
                        prior_visual_tokens=(
                            None
                            if not visual_blocks
                            else torch.cat(visual_blocks, dim=0)
                        ),
                        image_grid_thw=image_grid_thw,
                        image_features=image_features,
                    )

                current_visual_tokens, block_nfe = integrate_flow_tokens(
                    velocity,
                    initial,
                    num_steps=flow_steps,
                    solver=flow_solver,
                    state_dtype=flow_state_dtype,
                )
                flow_nfe += block_nfe
                current_visual_tokens = ablate_generated_latent_tokens(
                    current_visual_tokens,
                    mode=generated_latent_ablation,
                    seed=generated_latent_ablation_seed + block_index,
                )
            visual_blocks.append(current_visual_tokens)
            if use_generated_pixel_consumer:
                if capture_generated_pixels:
                    consumer_tokens, generated_pixels = (
                        model.generated_pixel_consumer_tokens(
                            current_visual_tokens,
                            pixel_ablation=pixel_ablation,
                            return_pixels=True,
                        )
                    )
                    generated_pixel_images.extend(
                        image.detach().float().cpu() for image in generated_pixels
                    )
                else:
                    consumer_tokens = model.generated_pixel_consumer_tokens(
                        current_visual_tokens,
                        pixel_ablation=pixel_ablation,
                    )
            else:
                consumer_tokens = current_visual_tokens
            consumer_visual_blocks.append(consumer_tokens)
            end_id = torch.tensor([[latent_end_id]], dtype=input_ids.dtype, device=device)
            input_ids = torch.cat((input_ids, end_id), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
                ),
                dim=1,
            )
            generated_ids.append(latent_end_id)
            continue

        if emitted_id in eos_ids:
            break
        if _ends_with_any(generated_ids, stop_token_sequences):
            break

    if diagnostics is not None:
        diagnostics.update(
            latent_blocks=latent_blocks,
            latent_steps=latent_size * latent_blocks,
            flow_steps=flow_steps * latent_blocks if oracle_latents is None else 0,
            flow_nfe=flow_nfe,
            flow_solver=flow_solver,
            flow_seed=flow_seed,
            flow_noise_device=flow_noise_device,
            flow_state_dtype=str(flow_state_dtype).removeprefix("torch."),
            flow_source_mode=flow_source_mode,
            ce_consumer_source=ce_consumer_source,
            visual_consumer_source=active_visual_consumer,
            generated_latent_ablation=generated_latent_ablation,
            generated_latent_ablation_seed=generated_latent_ablation_seed,
            flow_source_injected=(
                flow_source_latents is not None
                and bool(visual_blocks)
                and oracle_latents is None
            ),
            oracle_injected=oracle_latents is not None and bool(visual_blocks),
            oracle_blocks_forced_at_start=force_oracle_blocks_at_start,
        )
        if capture_generated_pixels:
            diagnostics["generated_pixel_images"] = generated_pixel_images
        if capture_generated_latents:
            diagnostics["generated_latent_blocks"] = [
                block.detach().cpu() for block in visual_blocks
            ]
    return input_ids
