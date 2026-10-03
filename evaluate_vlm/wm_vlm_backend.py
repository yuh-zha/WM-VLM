"""Evaluator backend for WM-VLM checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from wm_vlm.constants import (
    BOTTOM_LAYER_PLACEMENT,
    GAUSSIAN_NOISE_FLOW_SOURCE,
    QUERY_IMAGE_EMBEDDING_FLOW_SOURCE,
    SUPPORTED_VARIANTS,
    resolve_ce_consumer_source,
    resolve_flow_source_mode,
    resolve_layer_placement,
    resolve_routing_mode,
)
from wm_vlm.configuration_wm_vlm import WMVLMConfig
from wm_vlm.inference import (
    encode_flow_source_image_latents,
    encode_oracle_image_latents,
    generate_wm_vlm_answer,
)
from wm_vlm.modeling import (
    WMVLMForConditionalGeneration,
)
from wm_vlm.tokens import validate_wm_vlm_checkpoint


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
        raise ValueError(f"Unsupported WM-VLM dtype: {name}") from error


class WMVLMBackend:
    """Sequential evaluator with checkpoint-sized parallel flow sampling."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 128,
        max_latent_steps: int = 8,
        flow_solver: str = "heun",
        flow_seed: int = 0,
        flow_noise_device: str = "model",
        flow_state_dtype: str = "float32",
        top_p: float = 1.0,
        dtype: str = "bfloat16",
        device: str = "cuda:0",
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = False,
        allow_stage1_latent_inference: bool = False,
        save_generated_pixels_dir: str | Path | None = None,
        visual_consumer_mode: str = "checkpoint",
        generated_latent_ablation: str = "none",
        generated_latent_ablation_seed: int = 0,
        force_oracle_blocks_at_start: bool = False,
    ) -> None:
        del allow_stage1_latent_inference
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        # Preserve the existing evaluator CLI: max_latent_steps now denotes
        # ODE solver steps; the checkpoint separately fixes visual-token count.
        self.flow_steps = int(max_latent_steps)
        if self.flow_steps < 1:
            raise ValueError("Flow steps must be positive")
        if flow_solver not in {"euler", "heun"}:
            raise ValueError(
                f"Unsupported flow solver {flow_solver!r}; expected 'euler' or 'heun'"
            )
        self.flow_solver = flow_solver
        self.flow_seed = int(flow_seed)
        if flow_noise_device not in {"cpu", "model"}:
            raise ValueError(
                "flow_noise_device must be 'cpu' or 'model'; "
                f"got {flow_noise_device!r}"
            )
        if flow_state_dtype not in {"bfloat16", "float32"}:
            raise ValueError(
                "flow_state_dtype must be 'bfloat16' or 'float32'; "
                f"got {flow_state_dtype!r}"
            )
        self.flow_noise_device = flow_noise_device
        self.flow_state_dtype = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[flow_state_dtype]
        self.flow_state_dtype_name = flow_state_dtype
        self.top_p = float(top_p)
        supported_consumer_modes = {
            "checkpoint",
            "generated_pixel",
            "white_pixel",
            "shuffle_pixel_patches",
            "generated_endpoint",
        }
        if visual_consumer_mode not in supported_consumer_modes:
            raise ValueError(
                "visual_consumer_mode must be one of "
                f"{sorted(supported_consumer_modes)}, got {visual_consumer_mode!r}"
            )
        self.visual_consumer_mode = visual_consumer_mode
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
        self.generated_latent_ablation = generated_latent_ablation
        self.generated_latent_ablation_seed = int(generated_latent_ablation_seed)
        self.force_oracle_blocks_at_start = bool(force_oracle_blocks_at_start)
        self.num_generations = 0
        self.latent_activation_count = 0
        self.total_latent_steps = 0
        self.oracle_injection_count = 0
        self.total_oracle_latent_steps = 0
        self.flow_source_injection_count = 0
        self.save_generated_pixels_dir = (
            Path(save_generated_pixels_dir).resolve()
            if save_generated_pixels_dir is not None
            else None
        )
        if self.save_generated_pixels_dir is not None:
            self.save_generated_pixels_dir.mkdir(parents=True, exist_ok=True)
        self.last_generated_pixel_records: list[list[dict[str, object]]] = []
        loaded_config = AutoConfig.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
        )
        if loaded_config.model_type != WMVLMConfig.model_type:
            raise ValueError(
                "WM-VLM inference requires a native model_type='wm_vlm' "
                "checkpoint."
            )
        config = (
            loaded_config
            if isinstance(loaded_config, WMVLMConfig)
            else WMVLMConfig(**loaded_config.to_dict())
        )
        self.model = WMVLMForConditionalGeneration.from_pretrained(
            model,
            config=config,
            torch_dtype=_torch_dtype(dtype),
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        if getattr(self.model.config, "wm_vlm_variant", None) not in SUPPORTED_VARIANTS:
            raise ValueError(
                "Checkpoint is not a supported WM-VLM variant: "
                f"wm_vlm_variant={getattr(self.model.config, 'wm_vlm_variant', None)!r}"
            )
        self.routing_mode = resolve_routing_mode(self.model.config)
        self.layer_placement = resolve_layer_placement(self.model.config)
        self.flow_source_mode = resolve_flow_source_mode(self.model.config)
        self.ce_consumer_source = resolve_ce_consumer_source(self.model.config)
        self.requires_flow_source_image = (
            self.flow_source_mode == QUERY_IMAGE_EMBEDDING_FLOW_SOURCE
        )
        self.latent_size = int(self.model.config.wm_vlm_latent_size)
        if self.latent_size < 1:
            raise ValueError("WM-VLM evaluation requires a positive latent size")
        self.model.to(torch.device(device))
        self.model.eval()
        processor_use_fast = bool(
            getattr(self.model.config, "wm_vlm_processor_use_fast", False)
        )
        self.processor = AutoProcessor.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            use_fast=processor_use_fast,
        )
        validate_wm_vlm_checkpoint(self.model, self.processor.tokenizer)
        helper_min_pixels = getattr(self.model.config, "wm_vlm_helper_min_pixels", None)
        helper_max_pixels = getattr(self.model.config, "wm_vlm_helper_max_pixels", None)
        if helper_min_pixels is None or helper_max_pixels is None:
            self.helper_image_processor = self.processor.image_processor
        else:
            self.helper_image_processor = AutoProcessor.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                use_fast=processor_use_fast,
                min_pixels=int(helper_min_pixels),
                max_pixels=int(helper_max_pixels),
            ).image_processor
        source_min_pixels = getattr(
            self.model.config,
            "wm_vlm_flow_source_min_pixels",
            helper_min_pixels,
        )
        source_max_pixels = getattr(
            self.model.config,
            "wm_vlm_flow_source_max_pixels",
            helper_max_pixels,
        )
        if source_min_pixels is None or source_max_pixels is None:
            self.flow_source_image_processor = self.helper_image_processor
        else:
            self.flow_source_image_processor = AutoProcessor.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                use_fast=processor_use_fast,
                min_pixels=int(source_min_pixels),
                max_pixels=int(source_max_pixels),
            ).image_processor
        self.checkpoint_stage = f"wm_vlm_flow_{self.routing_mode}"
        if self.layer_placement == BOTTOM_LAYER_PLACEMENT:
            self.checkpoint_stage += "_bottom"
        elif self.layer_placement != "top":
            self.checkpoint_stage += f"_{self.layer_placement}"

    def generate(self, prompts: list[str], images: list[list[Image.Image]]) -> list[str]:
        return self._generate(
            prompts,
            images,
            oracle_images=None,
            flow_source_images=None,
        )

    def generate_with_flow_source(
        self,
        prompts: list[str],
        images: list[list[Image.Image]],
        flow_source_images: list[Image.Image],
    ) -> list[str]:
        return self._generate(
            prompts,
            images,
            oracle_images=None,
            flow_source_images=flow_source_images,
        )

    def generate_with_oracle(
        self,
        prompts: list[str],
        images: list[list[Image.Image]],
        oracle_images: list[Image.Image | list[Image.Image]],
    ) -> list[str]:
        return self._generate(
            prompts,
            images,
            oracle_images=oracle_images,
            flow_source_images=None,
        )

    def _generate(
        self,
        prompts: list[str],
        images: list[list[Image.Image]],
        *,
        oracle_images: list[Image.Image | list[Image.Image]] | None,
        flow_source_images: list[Image.Image] | None,
    ) -> list[str]:
        if len(prompts) != len(images):
            raise ValueError("Prompt and image batch lengths must match")
        if oracle_images is not None and len(oracle_images) != len(prompts):
            raise ValueError("Oracle image count must match prompt count")
        if flow_source_images is not None and len(flow_source_images) != len(prompts):
            raise ValueError("Flow-source image count must match prompt count")
        if oracle_images is not None and flow_source_images is not None:
            raise ValueError("Pass oracle images or flow-source images, not both")
        if (
            oracle_images is None
            and self.flow_source_mode == QUERY_IMAGE_EMBEDDING_FLOW_SOURCE
            and flow_source_images is None
        ):
            raise ValueError(
                "This checkpoint requires one query-crop flow-source image per prompt"
            )
        if (
            self.flow_source_mode == GAUSSIAN_NOISE_FLOW_SOURCE
            and flow_source_images is not None
        ):
            raise ValueError(
                "Gaussian-flow checkpoints cannot receive flow-source images"
            )
        predictions: list[str] = []
        self.last_generated_pixel_records = []
        for index, (question, image_list) in enumerate(zip(prompts, images, strict=True)):
            oracle_latents = None
            if oracle_images is not None:
                sample_oracle_images = oracle_images[index]
                if isinstance(sample_oracle_images, Image.Image):
                    sample_oracle_images = [sample_oracle_images]
                if not sample_oracle_images:
                    raise ValueError(
                        f"Prompt {index} must have at least one oracle image"
                    )
                oracle_latents = torch.cat(
                    [
                        encode_oracle_image_latents(
                            self.model,
                            self.processor,
                            oracle_image,
                            image_processor=self.helper_image_processor,
                        )
                        for oracle_image in sample_oracle_images
                    ],
                    dim=0,
                )
            flow_source_latents = (
                encode_flow_source_image_latents(
                    self.model,
                    self.flow_source_image_processor,
                    flow_source_images[index],
                )
                if flow_source_images is not None
                else None
            )
            diagnostics: dict[str, object] = {}
            predictions.append(
                generate_wm_vlm_answer(
                    self.model,
                    self.processor,
                    question=question,
                    images=image_list,
                    max_new_tokens=self.max_tokens,
                    flow_steps=self.flow_steps,
                    flow_solver=self.flow_solver,
                    flow_seed=self.flow_seed,
                    flow_noise_device=self.flow_noise_device,
                    flow_state_dtype=self.flow_state_dtype,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    oracle_latents=oracle_latents,
                    max_latent_blocks=(
                        int(oracle_latents.shape[0])
                        if oracle_latents is not None
                        else None
                    ),
                    flow_source_latents=flow_source_latents,
                    visual_consumer_mode=self.visual_consumer_mode,
                    generated_latent_ablation=self.generated_latent_ablation,
                    generated_latent_ablation_seed=(
                        self.generated_latent_ablation_seed + self.num_generations
                    ),
                    capture_generated_pixels=self.save_generated_pixels_dir is not None,
                    force_oracle_blocks_at_start=self.force_oracle_blocks_at_start,
                    diagnostics=diagnostics,
                )
            )
            pixel_records: list[dict[str, object]] = []
            generated_pixel_images = diagnostics.pop("generated_pixel_images", [])
            if not isinstance(generated_pixel_images, list):
                raise TypeError("generated_pixel_images diagnostic must be a list")
            for block_index, pixels in enumerate(generated_pixel_images):
                if (
                    not isinstance(pixels, torch.Tensor)
                    or pixels.ndim != 3
                    or pixels.shape[0] != 3
                ):
                    raise ValueError(
                        "Captured generated pixels must have shape [3, height, width], "
                        f"got {type(pixels).__name__} {getattr(pixels, 'shape', None)}"
                    )
                assert self.save_generated_pixels_dir is not None
                output_path = self.save_generated_pixels_dir / (
                    f"generation_{self.num_generations:06d}_block_{block_index:02d}.png"
                )
                pixels_float = pixels.float()
                clipped = pixels_float.clamp(0.0, 1.0)
                image_uint8 = (
                    clipped.mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .contiguous()
                    .numpy()
                )
                Image.fromarray(image_uint8, mode="RGB").save(output_path)
                pixel_records.append(
                    {
                        "block_index": block_index,
                        "path": str(output_path),
                        "height": int(pixels.shape[1]),
                        "width": int(pixels.shape[2]),
                        "raw_min": float(pixels_float.min()),
                        "raw_max": float(pixels_float.max()),
                        "raw_mean": float(pixels_float.mean()),
                        "raw_std": float(pixels_float.std()),
                        "fraction_below_zero": float((pixels_float < 0).float().mean()),
                        "fraction_above_one": float((pixels_float > 1).float().mean()),
                        "png_clamped_to_zero_one": True,
                    }
                )
            self.last_generated_pixel_records.append(pixel_records)
            self.num_generations += 1
            latent_steps = int(diagnostics["latent_steps"])
            self.latent_activation_count += int(latent_steps > 0)
            self.total_latent_steps += latent_steps
            oracle_injected = bool(diagnostics["oracle_injected"])
            self.oracle_injection_count += int(oracle_injected)
            self.total_oracle_latent_steps += latent_steps if oracle_injected else 0
            self.flow_source_injection_count += int(
                bool(diagnostics["flow_source_injected"])
            )
        return predictions
