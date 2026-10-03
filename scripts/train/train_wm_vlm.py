#!/usr/bin/env python3
"""Train wm_vlm with parallel flow-matched visual tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from wm_vlm.modeling import (  # noqa: E402
    WMVLMForConditionalGeneration,
)
from wm_vlm.configuration_wm_vlm import WMVLMConfig  # noqa: E402
from wm_vlm.constants import (  # noqa: E402
    BRANCH_ONLY_ROUTING,
    CE_CONSUMER_SOURCES,
    FLOW_SOURCE_MODES,
    FULL_VISION_TOKENS_TARGET,
    GAUSSIAN_NOISE_FLOW_SOURCE,
    GENERATED_PIXEL_CE_CONSUMER,
    ORACLE_TARGET_CE_CONSUMER,
    QUERY_IMAGE_EMBEDDING_FLOW_SOURCE,
    BRANCH_ONLY_INFERENCE_CONSUMER,
    LAYER_PLACEMENTS,
    SUPPORTED_VARIANTS,
    TOP_LAYER_PLACEMENT,
    TOKEN_EXPERT_QKV_ROUTING,
    resolve_layer_placement,
    resolve_flow_source_mode,
    resolve_routing_mode,
    training_consumer_mode,
    variant_for,
)
from wm_vlm.tokens import configure_wm_vlm_model  # noqa: E402
from wm_vlm.data import (  # noqa: E402
    WMVLMDataCollator,
    WMVLMManifestDataset,
    preprocess_with_qwen_vl_utils,
)
from wm_vlm.tokens import add_wm_vlm_special_tokens  # noqa: E402


TASK_DEFAULTS = {
    "tetris_2d": {
        "manifest": "datasets/Tetris-2D/manifest_wm_vlm_train.jsonl",
        "expected_train_examples": 4_000,
        "minimum_input_images": 1,
        "exact_input_images": 1,
        "expected_source_format": "tetris_2d",
        "helper_image_role": "rotated_query",
        "image_size": 512,
    },
    "tetris_3d": {
        "manifest": "datasets/Tetris-3D/manifest_wm_vlm_train.jsonl",
        "expected_train_examples": 16_000,
        "minimum_input_images": 1,
        "exact_input_images": 1,
        "expected_source_format": "tetris_3d",
        "helper_image_role": "rotation_state",
        "image_size": 512,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=-1)
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), default="tetris_2d")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--expected-train-examples", type=int, default=None)
    parser.add_argument("--minimum-input-images", type=int, default=None)
    parser.add_argument("--exact-input-images", type=int, default=None)
    parser.add_argument("--expected-source-format", default=None)
    parser.add_argument("--helper-image-role", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--problem-min-pixels", type=int, default=None)
    parser.add_argument("--problem-max-pixels", type=int, default=None)
    parser.add_argument("--helper-min-pixels", type=int, default=None)
    parser.add_argument("--helper-max-pixels", type=int, default=None)

    parser.add_argument(
        "--wm-vlm-num-layers",
        type=int,
        default=4,
        help="Number of Qwen decoder layers that receive a visual generation expert.",
    )
    parser.add_argument(
        "--wm-vlm-layer-placement",
        choices=LAYER_PLACEMENTS,
        default=TOP_LAYER_PLACEMENT,
        help=(
            "Place generation experts in the first (bottom), centered (middle), "
            "or last (top) Qwen decoder layers."
        ),
    )
    parser.add_argument(
        "--latent-size",
        type=int,
        default=None,
        help=(
            "Optional assertion for the visual reasoning token count. By default it is "
            "inferred from the complete post-merger vision encoding of a ground-truth "
            "helper image. A different value is rejected; targets are never truncated "
            "or pooled."
        ),
    )
    parser.add_argument(
        "--flow-loss-weight",
        type=float,
        default=1.0,
        help="Weight of conditional flow-matching velocity MSE.",
    )
    parser.add_argument(
        "--pixel-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of RGB reconstruction MSE decoded from the predicted flow "
            "endpoint. A positive value enables the pixel MLP head."
        ),
    )
    parser.add_argument(
        "--ce-weight",
        type=float,
        default=0.1,
        help=(
            "Weight of a separate clean-oracle text CE pass. Set to zero for "
            "flow-only training and one decoder pass per batch."
        ),
    )
    parser.add_argument(
        "--ce-consumer-source",
        choices=CE_CONSUMER_SOURCES,
        default=ORACLE_TARGET_CE_CONSUMER,
        help=(
            "Input supplied to the text CE consumer: the clean oracle latent, "
            "the differentiable predicted endpoint, or generated pixels passed "
            "back through the Qwen vision encoder."
        ),
    )
    parser.add_argument(
        "--flow-noise-scale",
        type=float,
        default=1.0,
        help="Standard deviation of the Gaussian flow source distribution.",
    )
    parser.add_argument(
        "--flow-source-mode",
        choices=FLOW_SOURCE_MODES,
        default=GAUSSIAN_NOISE_FLOW_SOURCE,
        help=(
            "Use the established Gaussian source or initialize the flow from "
            "Qwen vision embeddings of a per-example query-shape crop."
        ),
    )
    parser.add_argument(
        "--flow-source-image-dir",
        default=None,
        help=(
            "Directory containing one query-shape crop per question image, "
            "matched by basename; required for query_image_embedding."
        ),
    )
    parser.add_argument(
        "--initialize-generation-from-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy each selected pretrained Qwen layer into its new visual branch.",
    )
    parser.add_argument(
        "--freeze-vlm-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze original Qwen/LM-head parameters while training the new branch.",
    )
    parser.add_argument(
        "--freeze-generation-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Freeze generation experts/time embedding/velocity head and stop CE "
            "gradient at a generated consumer endpoint."
        ),
    )
    parser.add_argument(
        "--freeze-vision-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--prompt-style",
        choices=["wm_vlm", "tetris", "tetris_exact"],
        default="wm_vlm",
    )
    parser.add_argument(
        "--token-style",
        choices=["wm_vlm", "tetris"],
        default="wm_vlm",
    )
    parser.add_argument(
        "--processor-use-fast",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--padding", choices=["longest", "max_length"], default="longest")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--qwen-vl-utils-preprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--fixed-flow-noise-seed",
        type=int,
        default=None,
        help="Reuse one fixed Gaussian flow source tensor on every training step.",
    )
    parser.add_argument(
        "--fixed-flow-timestep",
        type=float,
        default=None,
        help="Reuse one flow timestep on every training step.",
    )
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument(
        "--optim",
        choices=["adamw_torch_fused", "adamw_torch"],
        default="adamw_torch_fused",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--disable-tqdm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable Trainer progress bars, useful for compact batch-job logs.",
    )
    parser.add_argument("--save-strategy", choices=["steps", "no"], default="steps")
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Keep only the three most recent checkpoints by default.",
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument(
        "--dataloader-persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = TASK_DEFAULTS[args.task]
    for field in (
        "manifest",
        "expected_train_examples",
        "minimum_input_images",
        "exact_input_images",
        "expected_source_format",
        "helper_image_role",
        "image_size",
    ):
        if getattr(args, field) is None:
            setattr(args, field, defaults[field])
    if args.data_root is None:
        args.data_root = os.fspath(Path(args.manifest).parent)
    for prefix in ("problem", "helper"):
        minimum_name = f"{prefix}_min_pixels"
        maximum_name = f"{prefix}_max_pixels"
        minimum = getattr(args, minimum_name)
        maximum = getattr(args, maximum_name)
        if (minimum is None) != (maximum is None):
            raise ValueError(f"Specify --{prefix}-min-pixels and --{prefix}-max-pixels together")
        if minimum is None:
            minimum = maximum = args.image_size * args.image_size
            setattr(args, minimum_name, minimum)
            setattr(args, maximum_name, maximum)
        if minimum <= 0 or maximum <= 0 or minimum > maximum:
            raise ValueError(f"Invalid {prefix} pixel bounds: {minimum}:{maximum}")
    if args.wm_vlm_num_layers < 1:
        raise ValueError("wm-vlm-num-layers must be positive")
    variant_for(args.wm_vlm_layer_placement)
    if args.latent_size is not None and args.latent_size < 1:
        raise ValueError("latent-size must be positive")
    if args.flow_loss_weight < 0:
        raise ValueError("flow-loss-weight must be non-negative")
    if args.pixel_loss_weight < 0:
        raise ValueError("pixel-loss-weight must be non-negative")
    if args.flow_noise_scale <= 0:
        raise ValueError("flow-noise-scale must be positive")
    if args.flow_source_mode == QUERY_IMAGE_EMBEDDING_FLOW_SOURCE:
        if args.task != "tetris_2d":
            raise ValueError(
                "query_image_embedding flow sources are currently defined only "
                "for the Tetris query-shape task"
            )
        if args.flow_source_image_dir is None:
            raise ValueError(
                "query_image_embedding requires --flow-source-image-dir"
            )
        flow_source_dir = Path(args.flow_source_image_dir).resolve()
        if not flow_source_dir.is_dir():
            raise ValueError(
                f"flow-source image directory does not exist: {flow_source_dir}"
            )
        args.flow_source_image_dir = os.fspath(flow_source_dir)
    elif args.flow_source_image_dir is not None:
        raise ValueError(
            "--flow-source-image-dir is valid only with "
            "--flow-source-mode query_image_embedding"
        )
    if args.ce_weight < 0:
        raise ValueError("ce-weight must be non-negative")
    if (
        args.flow_loss_weight == 0
        and args.pixel_loss_weight == 0
        and args.ce_weight == 0
    ):
        raise ValueError(
            "At least one of flow-loss-weight, pixel-loss-weight, or ce-weight "
            "must be positive"
        )
    if args.fixed_flow_timestep is not None and not 0 <= args.fixed_flow_timestep <= 1:
        raise ValueError("fixed-flow-timestep must lie in [0, 1]")
    if args.flow_source_mode == GAUSSIAN_NOISE_FLOW_SOURCE:
        if (args.fixed_flow_noise_seed is None) != (args.fixed_flow_timestep is None):
            raise ValueError(
                "Gaussian flow requires --fixed-flow-noise-seed and "
                "--fixed-flow-timestep together"
            )
    else:
        if args.fixed_flow_noise_seed is not None:
            raise ValueError(
                "Image-embedding flow sources cannot receive "
                "--fixed-flow-noise-seed"
            )
    if args.per_device_train_batch_size < 1:
        raise ValueError("per-device-train-batch-size must be positive")
    if args.save_total_limit < 1:
        raise ValueError("save-total-limit must be positive")
    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 and --fp16")
    return args


def infer_full_helper_vision_token_count(
    helper_image_processor: Any,
    helper_image_path: str | Path,
    *,
    spatial_merge_size: int,
    qwen_vl_utils_preprocess: bool = False,
) -> int:
    """Infer the lossless reasoning width from one real helper image.

    The collator/model subsequently validate every batch against this width,
    so heterogeneous helper grids fail instead of being silently reshaped.
    """
    if spatial_merge_size < 1:
        raise ValueError(
            f"spatial_merge_size must be positive, got {spatial_merge_size}"
        )
    with Image.open(helper_image_path) as source:
        image = source.convert("RGB")
    if qwen_vl_utils_preprocess:
        image = preprocess_with_qwen_vl_utils(image)
    helper_batch = helper_image_processor(images=[image], return_tensors="pt")
    image_grid_thw = helper_batch.get("image_grid_thw")
    if not isinstance(image_grid_thw, torch.Tensor) or tuple(
        image_grid_thw.shape
    ) != (1, 3):
        shape = getattr(image_grid_thw, "shape", None)
        raise ValueError(
            "Helper processor must return image_grid_thw with shape [1, 3], "
            f"got {shape}"
        )
    grid_tokens = int(image_grid_thw[0].prod().item())
    merge_area = spatial_merge_size * spatial_merge_size
    if grid_tokens % merge_area:
        raise ValueError(
            f"Helper grid {image_grid_thw.tolist()} is not divisible by vision "
            f"merge area {merge_area}"
        )
    token_count = grid_tokens // merge_area
    if token_count < 1:
        raise ValueError(f"Helper vision encoder produced invalid token count {token_count}")
    return token_count


def _distributed_context() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("wm_vlm training requires a GPU node")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is invalid for {torch.cuda.device_count()} GPUs"
        )
    torch.cuda.set_device(local_rank)
    return world_size, global_rank


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _checkpoint_tensor(checkpoint: Path, key: str) -> torch.Tensor:
    """Read one safetensors value without materializing the whole checkpoint."""
    from safetensors import safe_open

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        if key not in weight_map:
            raise KeyError(f"Checkpoint is missing audited weight {key!r}: {checkpoint}")
        tensor_path = checkpoint / weight_map[key]
    else:
        tensor_path = checkpoint / "model.safetensors"
        if not tensor_path.is_file():
            raise FileNotFoundError(
                f"Resume audit requires a safetensors checkpoint: {checkpoint}"
            )
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise KeyError(f"Checkpoint shard is missing audited weight {key!r}: {tensor_path}")
        return handle.get_tensor(key)


def audit_conversion_aware_resume(
    model: WMVLMForConditionalGeneration,
    checkpoint: str | Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Prove that renamed generation weights were restored exactly.

    The audit deliberately checks weights whose checkpoint names require Qwen's
    conversion mapping. A raw Trainer state-dict resume cannot pass this check.
    """
    checkpoint = Path(checkpoint).resolve()
    layer_indices = tuple(
        int(index)
        for index in getattr(model.config, "wm_vlm_layer_indices", ())
    )
    if not layer_indices:
        raise ValueError("Resume audit requires saved WM-VLM layer indices")
    first_layer = layer_indices[0]
    last_layer = layer_indices[-1]
    candidates = [
        (
            f"model.layers.{first_layer}.generation_self_attn.q_proj.weight",
            f"model.language_model.layers.{first_layer}.generation_self_attn.q_proj.weight",
        ),
        (
            f"model.layers.{last_layer}.generation_mlp.down_proj.weight",
            f"model.language_model.layers.{last_layer}.generation_mlp.down_proj.weight",
        ),
        ("flow_velocity_head.weight", "flow_velocity_head.weight"),
    ]
    if hasattr(model, "generation_final_norm"):
        candidates.append(
            ("generation_final_norm.weight", "generation_final_norm.weight")
        )

    model_state = model.state_dict()
    records: list[dict[str, Any]] = []
    for checkpoint_key, runtime_key in candidates:
        if runtime_key not in model_state:
            raise KeyError(f"Runtime model is missing audited weight {runtime_key!r}")
        saved = _checkpoint_tensor(checkpoint, checkpoint_key)
        loaded = model_state[runtime_key].detach().cpu()
        exact_match = bool(torch.equal(saved, loaded))
        record = {
            "checkpoint_key": checkpoint_key,
            "runtime_key": runtime_key,
            "shape": list(saved.shape),
            "dtype": str(saved.dtype),
            "checkpoint_sha256": _tensor_sha256(saved),
            "runtime_sha256": _tensor_sha256(loaded),
            "exact_match": exact_match,
        }
        records.append(record)
        if not exact_match:
            raise RuntimeError(
                "Conversion-aware resume audit failed for "
                f"{checkpoint_key!r} -> {runtime_key!r}"
            )

    payload = {
        "checkpoint": os.fspath(checkpoint),
        "loader": "WMVLMForConditionalGeneration.from_pretrained",
        "conversion_mapping": dict(model._checkpoint_conversion_mapping),
        "all_exact_match": True,
        "records": records,
    }
    audit_dir = output_dir / "resume_load_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / f"{checkpoint.name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


class FlowMetricsTrainer(Trainer):
    """Aggregate flow and clean-text diagnostics over microbatches and ranks."""

    COMPONENTS = (
        "flow_loss",
        "pixel_loss",
        "ce_loss",
        "endpoint_mse",
        "endpoint_cosine",
        "predicted_velocity_norm",
        "target_velocity_norm",
        "target_latent_norm",
        "flow_t_mean",
    )

    def __init__(
        self,
        *args,
        metrics_path: Path | None = None,
        fixed_flow_noise_seed: int | None = None,
        fixed_flow_timestep: float | None = None,
        preloaded_resume_checkpoint: str | Path | None = None,
        **kwargs,
    ) -> None:
        self.preloaded_resume_checkpoint = (
            None
            if preloaded_resume_checkpoint is None
            else Path(preloaded_resume_checkpoint).resolve()
        )
        self.metrics_path = metrics_path
        self.component_sums: dict[str, torch.Tensor] = {}
        self.component_count = 0
        self.fixed_flow_timestep = fixed_flow_timestep
        super().__init__(*args, **kwargs)
        self.flow_source_mode = resolve_flow_source_mode(self.model.config)

        def make_flow_noise(seed: int | None) -> torch.Tensor | None:
            if seed is None:
                return None
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            return torch.randn(
                (
                    1,
                    int(self.model.config.wm_vlm_latent_size),
                    int(self.model.config.text_config.hidden_size),
                ),
                generator=generator,
                dtype=torch.float32,
            ) * float(self.model.config.wm_vlm_flow_noise_scale)

        self.fixed_flow_noise = make_flow_noise(fixed_flow_noise_seed)
        if self.flow_source_mode == GAUSSIAN_NOISE_FLOW_SOURCE:
            if (self.fixed_flow_noise is None) != (self.fixed_flow_timestep is None):
                raise ValueError(
                    "fixed Gaussian flow noise and timestep must be configured together"
                )
        elif self.fixed_flow_noise is not None:
            raise ValueError(
                "Image-embedding flow sources cannot be combined with Gaussian noise"
            )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Skip Trainer's conversion-unaware model reload after a preload.

        Qwen2.5-VL ``save_pretrained`` writes compatibility keys such as
        ``visual.*`` and ``model.layers.*``. Transformers Trainer restores a
        checkpoint with a raw state-dict load, which does not apply Qwen's
        ``_checkpoint_conversion_mapping`` back to ``model.visual.*`` and
        ``model.language_model.layers.*``. The caller therefore constructs the
        model from the resume checkpoint through ``from_pretrained`` first.
        Trainer still restores optimizer, scheduler, scaler, RNG, and global
        step state after this method returns.
        """
        requested = Path(resume_from_checkpoint).resolve()
        if self.preloaded_resume_checkpoint is None:
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        if requested != self.preloaded_resume_checkpoint:
            raise RuntimeError(
                "Trainer resume checkpoint differs from the conversion-aware "
                f"model preload: requested={requested}, "
                f"preloaded={self.preloaded_resume_checkpoint}"
            )
        print(
            "[wm_vlm] model weights already loaded with "
            f"conversion-aware from_pretrained: {requested}",
            flush=True,
        )
        return None

    def _apply_fixed_flow(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        noise = self.fixed_flow_noise
        timestep = self.fixed_flow_timestep
        if timestep is None:
            return inputs
        batch_size = int(inputs["input_ids"].shape[0])
        block_counts = inputs.get("latent_block_counts")
        flow_items = (
            batch_size
            if block_counts is None
            else int(block_counts.detach().cpu().sum().item())
        )
        conditioned = dict(inputs)
        if self.flow_source_mode == GAUSSIAN_NOISE_FLOW_SOURCE:
            if noise is None:
                raise RuntimeError("Fixed Gaussian timestep is missing its flow noise")
            noise = noise.to(device=inputs["input_ids"].device)
            if flow_items != 1:
                noise = noise.expand(flow_items, -1, -1)
            conditioned["flow_noise"] = noise
        conditioned["flow_timesteps"] = torch.full(
            (flow_items,),
            timestep,
            dtype=torch.float32,
            device=inputs["input_ids"].device,
        )
        return conditioned

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        inputs = self._apply_fixed_flow(inputs)
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        values: dict[str, torch.Tensor] = {}
        for name in self.COMPONENTS:
            value = getattr(outputs, name, None)
            if value is not None:
                values[name] = value.detach().float()
        for name, value in values.items():
            self.component_sums[name] = self.component_sums.get(
                name,
                torch.zeros_like(value),
            ) + value
        self.component_count += 1
        return (loss, outputs) if return_outputs else loss

    def _flush_components(self) -> dict[str, float]:
        if not self.component_count or not self.component_sums:
            return {}
        names = sorted(self.component_sums)
        device = next(iter(self.component_sums.values())).device
        packed = torch.stack(
            [self.component_sums[name] for name in names]
            + [torch.tensor(float(self.component_count), device=device)]
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        count = packed[-1].clamp_min(1.0)
        metrics = {
            name: float((packed[index] / count).item())
            for index, name in enumerate(names)
        }
        self.component_sums.clear()
        self.component_count = 0
        return metrics

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        logs = dict(logs)
        components = self._flush_components()
        logs.update(components)
        super().log(logs, start_time)
        if components and self.metrics_path is not None and self.is_world_process_zero():
            record = {"step": self.state.global_step, "epoch": self.state.epoch, **components}
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _training_arguments(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    report_to = [] if args.report_to.lower() in {"", "none"} else [args.report_to]
    return TrainingArguments(
        output_dir=os.fspath(output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        logging_steps=args.logging_steps,
        disable_tqdm=args.disable_tqdm,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        report_to=report_to,
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
        save_safetensors=True,
    )


def _write_run_config(
    args: argparse.Namespace,
    output_dir: Path,
    dataset: WMVLMManifestDataset,
    *,
    world_size: int,
) -> None:
    manifest_path = Path(args.manifest)
    digest = hashlib.sha256()
    with manifest_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    payload = vars(args).copy()
    payload.update(
        {
            "variant": variant_for(args.wm_vlm_layer_placement),
            "routing_mode": BRANCH_ONLY_ROUTING,
            "wm_vlm_layer_placement": args.wm_vlm_layer_placement,
            "training_consumer_mode": training_consumer_mode(
                args.ce_consumer_source,
            ),
            "inference_consumer_mode": BRANCH_ONLY_INFERENCE_CONSUMER,
            "latent_size": args.latent_size,
            "target_mode": FULL_VISION_TOKENS_TARGET,
            "parallel_visual_tokens": True,
            "qkv_routing": TOKEN_EXPERT_QKV_ROUTING,
            "attention_path": (
                "per-token modality Q/K/V projections + one global self-attention"
            ),
            "flow_path": (
                f"x_t=(1-t)*{args.flow_source_mode}+t*target; "
                f"velocity=target-{args.flow_source_mode}"
            ),
            "flow_source": args.flow_source_mode,
            "target": "all post-merger Qwen vision encoder tokens (no pooling/truncation)",
            "manifest_sha256": digest.hexdigest(),
            "eligible_train_examples": len(dataset),
            "supervised_latent_blocks": getattr(
                dataset,
                "total_helper_images",
                len(dataset),
            ),
            "max_latent_blocks_per_example": getattr(
                dataset,
                "max_helper_images_per_example",
                1,
            ),
            "latent_block_training": (
                "causal_ground_truth_prior_block_teacher_forcing"
                if getattr(dataset, "max_helper_images_per_example", 1) > 1
                else "single_block"
            ),
            "world_size": world_size,
            "effective_global_batch_size": (
                world_size
                * args.per_device_train_batch_size
                * args.gradient_accumulation_steps
            ),
            "loss_formula": (
                f"{args.flow_loss_weight}*flow_mse + "
                f"{args.pixel_loss_weight}*pixel_mse + "
                f"{args.ce_weight}*{args.ce_consumer_source}_ce"
            ),
        }
    )
    (output_dir / "wm_vlm_run_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = resolve_args(parse_args())
    world_size, global_rank = _distributed_context()
    is_main = global_rank == 0
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_load_source = args.resume_from_checkpoint or args.model

    processor = AutoProcessor.from_pretrained(
        model_load_source,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.processor_use_fast,
        min_pixels=args.problem_min_pixels,
        max_pixels=args.problem_max_pixels,
    )
    helper_processor = AutoProcessor.from_pretrained(
        model_load_source,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.processor_use_fast,
        min_pixels=args.helper_min_pixels,
        max_pixels=args.helper_max_pixels,
    ).image_processor
    config = WMVLMConfig.from_qwen_config(
        AutoConfig.from_pretrained(
            model_load_source,
            trust_remote_code=args.trust_remote_code,
        )
    )
    args.pixel_patch_size = (
        int(config.vision_config.patch_size)
        * int(config.vision_config.spatial_merge_size)
    )
    configured_pixel_patch_size = getattr(
        config,
        "wm_vlm_pixel_patch_size",
        None,
    )
    if (
        configured_pixel_patch_size is not None
        and int(configured_pixel_patch_size) != args.pixel_patch_size
    ):
        raise ValueError(
            "Checkpoint pixel patch size does not match its Qwen vision geometry: "
            f"checkpoint={configured_pixel_patch_size}, inferred={args.pixel_patch_size}"
        )
    pixel_path_enabled = (
        args.pixel_loss_weight > 0
        or args.ce_consumer_source == GENERATED_PIXEL_CE_CONSUMER
    )
    if pixel_path_enabled:
        config.wm_vlm_pixel_reconstruction = True
        config.wm_vlm_pixel_patch_size = args.pixel_patch_size
    source_dataset = WMVLMManifestDataset(
        args.manifest,
        root_dir=args.data_root,
        expected_examples=args.expected_train_examples,
        minimum_input_images=args.minimum_input_images,
        exact_input_images=args.exact_input_images,
        expected_source_format=args.expected_source_format,
        task_name=args.task,
        flow_source_image_dir=args.flow_source_image_dir,
    )
    dataset = source_dataset
    full_vision_token_count = infer_full_helper_vision_token_count(
        helper_processor,
        dataset[0].helper_image_path,
        spatial_merge_size=int(config.vision_config.spatial_merge_size),
        qwen_vl_utils_preprocess=args.qwen_vl_utils_preprocess,
    )
    if args.latent_size is None:
        args.latent_size = full_vision_token_count
    elif args.latent_size != full_vision_token_count:
        raise ValueError(
            "--latent-size is now only an assertion for the complete helper vision "
            f"token count: requested={args.latent_size}, encoded="
            f"{full_vision_token_count}. Remove the flag to infer it automatically. "
            "Truncation and pooling are disabled."
        )
    if args.flow_source_mode == QUERY_IMAGE_EMBEDDING_FLOW_SOURCE:
        flow_source_path = dataset[0].flow_source_image_path
        if flow_source_path is None:
            raise RuntimeError("Dataset did not attach the configured flow-source image")
        flow_source_token_count = infer_full_helper_vision_token_count(
            helper_processor,
            flow_source_path,
            spatial_merge_size=int(config.vision_config.spatial_merge_size),
            qwen_vl_utils_preprocess=args.qwen_vl_utils_preprocess,
        )
        if flow_source_token_count != args.latent_size:
            raise ValueError(
                "Query-crop flow-source token count must match the helper target: "
                f"source={flow_source_token_count}, target={args.latent_size}"
            )
    added_tokens = add_wm_vlm_special_tokens(
        processor.tokenizer,
        token_style=args.token_style,
    )
    is_wm_vlm_checkpoint = (
        getattr(config, "wm_vlm_variant", None) in SUPPORTED_VARIANTS
    )
    if is_wm_vlm_checkpoint:
        resolve_routing_mode(config)
        saved_layer_placement = resolve_layer_placement(config)
        if saved_layer_placement != args.wm_vlm_layer_placement:
            raise ValueError(
                "Checkpoint wm_vlm layer placement does not match "
                "--wm-vlm-layer-placement: "
                f"checkpoint={saved_layer_placement}, "
                f"requested={args.wm_vlm_layer_placement}"
            )
        saved_layers = int(getattr(config, "wm_vlm_num_layers", -1))
        if saved_layers != args.wm_vlm_num_layers:
            raise ValueError(
                "Checkpoint wm_vlm depth does not match --wm-vlm-num-layers: "
                f"checkpoint={saved_layers}, requested={args.wm_vlm_num_layers}"
            )
        saved_latent_size = int(
            getattr(config, "wm_vlm_latent_size", -1)
        )
        if saved_latent_size != args.latent_size:
            raise ValueError(
                "Checkpoint latent size does not match --latent-size: "
                f"checkpoint={saved_latent_size}, requested={args.latent_size}"
            )
        saved_flow_source_mode = resolve_flow_source_mode(config)
        if saved_flow_source_mode != args.flow_source_mode:
            raise ValueError(
                "Checkpoint flow source does not match --flow-source-mode: "
                f"checkpoint={saved_flow_source_mode}, requested={args.flow_source_mode}"
            )
        saved_generation_initialization = getattr(
            config,
            "wm_vlm_generation_initialization",
            None,
        )
        requested_generation_initialization = (
            "corresponding_text_layers"
            if args.initialize_generation_from_text
            else "random"
        )
        if (
            saved_generation_initialization is not None
            and saved_generation_initialization
            != requested_generation_initialization
        ):
            raise ValueError(
                "Checkpoint generation-branch initialization does not match "
                "--initialize-generation-from-text: "
                f"checkpoint={saved_generation_initialization}, "
                f"requested={requested_generation_initialization}"
            )
        if added_tokens:
            raise ValueError("WM-VLM checkpoint processor is missing its saved special tokens")
    else:
        config.wm_vlm_num_layers = int(args.wm_vlm_num_layers)
        config.wm_vlm_latent_size = int(args.latent_size)
        config.wm_vlm_target_mode = FULL_VISION_TOKENS_TARGET
        config.wm_vlm_routing_mode = BRANCH_ONLY_ROUTING
        config.wm_vlm_layer_placement = args.wm_vlm_layer_placement
        config.wm_vlm_qkv_routing = TOKEN_EXPERT_QKV_ROUTING
        config.text_config.wm_vlm_qkv_routing = TOKEN_EXPERT_QKV_ROUTING

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    model = WMVLMForConditionalGeneration.from_pretrained(
        model_load_source,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    if not is_wm_vlm_checkpoint:
        model.resize_token_embeddings(len(processor.tokenizer))
        if args.initialize_generation_from_text:
            model.initialize_generation_branch_from_text()
    elif model.get_input_embeddings().weight.shape[0] != len(processor.tokenizer):
        raise ValueError("Checkpoint embedding rows do not match its tokenizer")
    if is_main and args.resume_from_checkpoint is not None:
        audit = audit_conversion_aware_resume(
            model,
            args.resume_from_checkpoint,
            output_dir,
        )
        print(
            "[wm_vlm] conversion-aware resume audit passed "
            f"checkpoint={args.resume_from_checkpoint} "
            f"tensors={len(audit['records'])}",
            flush=True,
        )

    configure_wm_vlm_model(
        model,
        processor.tokenizer,
        wm_vlm_num_layers=args.wm_vlm_num_layers,
        latent_size=args.latent_size,
        flow_loss_weight=args.flow_loss_weight,
        pixel_loss_weight=args.pixel_loss_weight,
        pixel_patch_size=args.pixel_patch_size,
        ce_weight=args.ce_weight,
        ce_consumer_source=args.ce_consumer_source,
        flow_noise_scale=args.flow_noise_scale,
        flow_source_mode=args.flow_source_mode,
        freeze_vision=args.freeze_vision_encoder,
        token_style=args.token_style,
        layer_placement=args.wm_vlm_layer_placement,
        target_mode=FULL_VISION_TOKENS_TARGET,
    )
    model.config.wm_vlm_generation_initialization = (
        "corresponding_text_layers"
        if args.initialize_generation_from_text
        else "random"
    )
    pixel_image_mean = tuple(float(value) for value in helper_processor.image_mean)
    pixel_image_std = tuple(float(value) for value in helper_processor.image_std)
    if len(pixel_image_mean) != 3 or len(pixel_image_std) != 3:
        raise ValueError(
            "Qwen helper processor must expose three-channel image mean/std"
        )
    model.config.wm_vlm_pixel_image_mean = list(pixel_image_mean)
    model.config.wm_vlm_pixel_image_std = list(pixel_image_std)
    model.config.wm_vlm_generated_pixel_preprocess = (
        "differentiable_qwen_normalize_temporal_duplicate_merge_patchify"
    )
    model.config.wm_vlm_task = args.task
    model.config.wm_vlm_image_size = int(args.image_size)
    model.config.wm_vlm_problem_min_pixels = int(args.problem_min_pixels)
    model.config.wm_vlm_problem_max_pixels = int(args.problem_max_pixels)
    model.config.wm_vlm_helper_min_pixels = int(args.helper_min_pixels)
    model.config.wm_vlm_helper_max_pixels = int(args.helper_max_pixels)
    model.config.wm_vlm_pixel_target_range = "zero_to_one"
    model.config.wm_vlm_helper_image_role = args.helper_image_role
    model.config.wm_vlm_flow_source_image_role = (
        "question_query_shape_crop"
        if args.flow_source_mode == QUERY_IMAGE_EMBEDDING_FLOW_SOURCE
        else None
    )
    model.config.wm_vlm_flow_source_min_pixels = args.helper_min_pixels
    model.config.wm_vlm_flow_source_max_pixels = args.helper_max_pixels
    model.config.wm_vlm_flow_source_qwen_vl_utils_preprocess = bool(
        args.qwen_vl_utils_preprocess
    )
    model.config.wm_vlm_prompt_style = args.prompt_style
    model.config.wm_vlm_token_style = args.token_style
    model.config.wm_vlm_processor_use_fast = bool(args.processor_use_fast)
    model.config.wm_vlm_vl_utils_preprocess = bool(args.qwen_vl_utils_preprocess)
    model.config.wm_vlm_problem_qwen_vl_utils_preprocess = bool(
        args.qwen_vl_utils_preprocess
    )
    model.config.wm_vlm_helper_qwen_vl_utils_preprocess = bool(
        args.qwen_vl_utils_preprocess
    )
    model.config.wm_vlm_freeze_vlm_backbone = bool(args.freeze_vlm_backbone)
    model.config.wm_vlm_freeze_generation_branch = bool(
        args.freeze_generation_branch
    )
    model.config.wm_vlm_multiblock_training = bool(
        source_dataset.max_helper_images_per_example > 1
    )
    model.config.wm_vlm_max_latent_blocks = int(
        source_dataset.max_helper_images_per_example
    )
    model.config.wm_vlm_latent_teacher_forcing = (
        "ground_truth_prior_blocks"
        if source_dataset.max_helper_images_per_example > 1
        else "not_applicable"
    )
    model.config.use_cache = False
    model.accepts_loss_kwargs = False

    model.set_vlm_backbone_trainable(not args.freeze_vlm_backbone)
    model.set_generation_branch_trainable(not args.freeze_generation_branch)
    model.set_vision_encoder_trainable(not args.freeze_vision_encoder)

    collator = WMVLMDataCollator(
        processor,
        stage="stage1",
        latent_size=args.latent_size,
        helper_image_processor=helper_processor,
        flow_source_image_processor=helper_processor,
        prompt_style=args.prompt_style,
        token_style=args.token_style,
        padding=args.padding,
        max_length=args.max_length,
        qwen_vl_utils_preprocess=args.qwen_vl_utils_preprocess,
        pixel_reconstruction=pixel_path_enabled,
        pixel_patch_size=args.pixel_patch_size,
        spatial_merge_size=int(config.vision_config.spatial_merge_size),
    )

    if is_main:
        processor.save_pretrained(output_dir)
        _write_run_config(args, output_dir, dataset, world_size=world_size)
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        branch = sum(
            parameter.numel()
            for layer in model.wm_vlm_layers
            for module in (
                layer.generation_self_attn,
                layer.generation_mlp,
                layer.generation_input_layernorm,
                layer.generation_post_attention_layernorm,
            )
            for parameter in module.parameters()
        )
        branch += sum(parameter.numel() for parameter in model.flow_time_embedding.parameters())
        branch += sum(parameter.numel() for parameter in model.flow_velocity_head.parameters())
        if hasattr(model, "pixel_reconstruction_head"):
            branch += sum(
                parameter.numel()
                for parameter in model.pixel_reconstruction_head.parameters()
            )
        if hasattr(model, "generation_final_norm"):
            branch += sum(
                parameter.numel()
                for parameter in model.generation_final_norm.parameters()
            )
        print(
            f"[{variant_for(args.wm_vlm_layer_placement)}] "
            f"task={args.task} examples={len(dataset)} wm_vlm_layers={args.wm_vlm_num_layers} "
            f"latent_blocks={source_dataset.total_helper_images} "
            f"max_blocks_per_example={source_dataset.max_helper_images_per_example} "
            f"routing={BRANCH_ONLY_ROUTING} "
            f"layer_placement={args.wm_vlm_layer_placement} "
            f"qkv_routing={TOKEN_EXPERT_QKV_ROUTING} "
            f"flow_source={args.flow_source_mode} "
            f"train_consumer={training_consumer_mode(args.ce_consumer_source)} "
            f"ce_consumer_source={args.ce_consumer_source} "
            f"inference_consumer={BRANCH_ONLY_INFERENCE_CONSUMER} "
            f"parallel_visual_tokens={args.latent_size} branch_parameters={branch:,} "
            f"trainable={trainable:,}/{total:,} world_size={world_size}",
            flush=True,
        )

    trainer = FlowMetricsTrainer(
        model=model,
        args=_training_arguments(args, output_dir),
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
        metrics_path=output_dir / "flow_component_metrics.jsonl",
        fixed_flow_noise_seed=args.fixed_flow_noise_seed,
        fixed_flow_timestep=args.fixed_flow_timestep,
        preloaded_resume_checkpoint=args.resume_from_checkpoint,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
