#!/usr/bin/env python3
"""Supervised fine-tuning controls for the paper's stock Qwen2.5-VL baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from wm_vlm.data import (
    WMVLMManifestDataset,
    QwenDirectSFTDataCollator,
    QwenImaginationSFTDataCollator,
    QwenImaginationWithHelperImageSFTDataCollator,
    QwenInterleavedImagePlaceholderSFTDataCollator,
)


TASK_DEFAULTS = {
    "tetris_2d": {
        "manifest": "datasets/Tetris-2D/manifest_wm_vlm_train.jsonl",
        "expected_train_examples": 4_000,
        "minimum_input_images": 1,
        "exact_input_images": 1,
        "expected_source_format": "tetris_2d",
        "image_size": 512,
        "frame_policy": (
            "use one composite problem image and supervise only the final option letter; "
            "ignore all intermediate reasoning text and helper images"
        ),
    },
    "tetris_3d": {
        "manifest": "datasets/Tetris-3D/manifest_wm_vlm_train.jsonl",
        "expected_train_examples": 16_000,
        "minimum_input_images": 1,
        "exact_input_images": 1,
        "expected_source_format": "tetris_3d",
        "image_size": 512,
        "frame_policy": (
            "retain the problem image and answer supervision while replacing each "
            "intermediate visual state with literal [IMAGINATION]"
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), default="tetris_2d")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--expected-train-examples", type=int, default=None)
    parser.add_argument("--minimum-input-images", type=int, default=None)
    parser.add_argument("--exact-input-images", type=int, default=None)
    parser.add_argument("--expected-source-format", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--supervision-mode",
        choices=[
            "answer_only",
            "interleaved_image_placeholder",
            "reasoning_imagination",
            "reasoning_imagination_with_helper_image",
        ],
        default="answer_only",
        help=(
            "Use answer-only supervision; ordinary interleaved SFT with each target "
            "image replaced in place by literal [IMAGINATION]; or an explicitly "
            "prompted imagination variant, optionally followed by the GT helper "
            "image as training-only context."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=8_400)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=["steps", "no"], default="steps")
    parser.add_argument("--save-steps", type=int, default=8_400)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--skip-final-save", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def _resolve_task_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = TASK_DEFAULTS[args.task]
    for key in (
        "manifest",
        "expected_train_examples",
        "minimum_input_images",
        "exact_input_images",
        "expected_source_format",
        "image_size",
    ):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    if args.data_root is None:
        args.data_root = os.fspath(Path(args.manifest).parent)
    return args


def _distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= global_rank < world_size:
        raise RuntimeError(
            f"Invalid distributed environment: WORLD_SIZE={world_size}, RANK={global_rank}"
        )
    if world_size > 1 and "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "WORLD_SIZE is greater than one but LOCAL_RANK is missing. Launch ordinary DDP "
            "with torchrun, not independent Python processes."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen fine-tuning must run on a GPU node")
    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is invalid for {torch.cuda.device_count()} visible GPUs"
        )
    torch.cuda.set_device(local_rank)
    return world_size, global_rank, local_rank


def _write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    dataset: WMVLMManifestDataset,
    *,
    world_size: int,
) -> None:
    reasoning_imagination = args.supervision_mode in {
        "interleaved_image_placeholder",
        "reasoning_imagination",
        "reasoning_imagination_with_helper_image",
    }
    ordinary_image_placeholder = (
        args.supervision_mode == "interleaved_image_placeholder"
    )
    uses_helper_image = (
        args.supervision_mode == "reasoning_imagination_with_helper_image"
    )
    payload = vars(args).copy()
    payload.update(
        {
            "method": (
                "stock_qwen_reasoning_imagination_with_gt_helper_train_sft"
                if uses_helper_image
                else (
                    "stock_qwen_ordinary_interleaved_image_placeholder_sft"
                    if ordinary_image_placeholder
                    else (
                        "stock_qwen_reasoning_imagination_sft"
                        if reasoning_imagination
                        else "stock_qwen_direct_answer_sft"
                    )
                )
            ),
            "target_format": (
                "full reasoning trace with literal [IMAGINATION] followed by the "
                "GT helper image as causal training context"
                if uses_helper_image
                else (
                    "manifest system prompt and full reasoning trace with each "
                    "supervised image replaced in place by literal [IMAGINATION]"
                    if ordinary_image_placeholder
                    else (
                        "full reasoning trace with supervised helper image replaced "
                        "by literal [IMAGINATION]"
                        if reasoning_imagination
                        else "assistant answer only"
                    )
                )
            ),
            "uses_wm_vlm_tokens": False,
            "uses_helper_image": uses_helper_image,
            "evaluation_uses_helper_image": False,
            "uses_reasoning_text": reasoning_imagination,
            "helper_image_replacement": (
                "[IMAGINATION]" if reasoning_imagination else None
            ),
            "helper_image_transform_time": (
                "collator_read_time" if reasoning_imagination else None
            ),
            "adds_imagination_instruction": (
                args.supervision_mode
                in {
                    "reasoning_imagination",
                    "reasoning_imagination_with_helper_image",
                }
            ),
            "uses_manifest_system_prompt": ordinary_image_placeholder,
            "assistant_image_token_ce_masked": uses_helper_image,
            "freeze_vision_encoder": True,
            "eligible_train_examples": len(dataset),
            "manifest_total_rows": dataset.total_rows,
            "skipped_short_rows": dataset.skipped_short_rows,
            "task": args.task,
            "expected_source_format": args.expected_source_format,
            "frame_policy": (
                "retain source/problem images and insert the GT helper image after "
                "literal [IMAGINATION] as training-only causal context"
                if uses_helper_image
                else (
                    "retain source/problem images and the complete reasoning/answer "
                    "trace; replace every supervised intermediate image in place "
                    "with literal [IMAGINATION]"
                    if ordinary_image_placeholder
                    else (
                        "retain source/problem images and replace the single "
                        "supervised helper image with literal [IMAGINATION] at "
                        "collator read time"
                        if reasoning_imagination
                        else TASK_DEFAULTS[args.task]["frame_policy"]
                    )
                )
            ),
            "distributed_strategy": "ddp" if world_size > 1 else "single_process",
            "world_size": world_size,
            "per_device_train_batch_size": 1,
            "effective_global_batch_size": world_size * args.gradient_accumulation_steps,
            "resume_model_loading": (
                "from_pretrained_conversion_aware_then_trainer_optimizer_scheduler_resume"
            ),
        }
    )
    (output_dir / "qwen_direct_sft_run_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class QwenConversionAwareResumeTrainer(Trainer):
    """Avoid Trainer's raw sharded load, which bypasses Qwen key conversion.

    Qwen2.5-VL ``save_pretrained`` writes compatibility keys such as ``visual.*``
    and ``model.layers.*``.  In Transformers 4.57, ``Trainer`` resumes sharded
    checkpoints with ``load_sharded_checkpoint`` and does not apply the model's
    ``_checkpoint_conversion_mapping``.  The caller therefore initializes the
    model from the checkpoint through ``from_pretrained`` first; this override
    skips only Trainer's duplicate raw model load.  Trainer still restores its
    optimizer, scheduler, scaler, RNG, and global-step state normally.
    """

    def __init__(self, *args, preloaded_resume_checkpoint: str | None = None, **kwargs):
        self.preloaded_resume_checkpoint = (
            None
            if preloaded_resume_checkpoint is None
            else Path(preloaded_resume_checkpoint).resolve()
        )
        super().__init__(*args, **kwargs)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        requested = Path(resume_from_checkpoint).resolve()
        if self.preloaded_resume_checkpoint is None:
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        if requested != self.preloaded_resume_checkpoint:
            raise RuntimeError(
                "Trainer resume checkpoint differs from the conversion-aware model "
                f"preload: requested={requested}, preloaded={self.preloaded_resume_checkpoint}"
            )
        print(
            "[qwen-sft] model weights already loaded with conversion-aware "
            f"from_pretrained: {requested}",
            flush=True,
        )
        return None


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _audit_conversion_aware_resume(
    model: Qwen2_5_VLForConditionalGeneration,
    checkpoint: str | Path,
    output_dir: Path,
) -> None:
    """Prove that one renamed language weight exactly matches the checkpoint."""
    from safetensors import safe_open

    checkpoint = Path(checkpoint).resolve()
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Resume audit requires a sharded safetensors index: {index_path}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    candidates = (
        (
            "model.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.input_layernorm.weight",
        ),
        ("visual.merger.ln_q.weight", "model.visual.merger.ln_q.weight"),
    )
    model_state = model.state_dict()
    records: list[dict[str, object]] = []
    for checkpoint_key, runtime_key in candidates:
        if checkpoint_key not in weight_map or runtime_key not in model_state:
            raise KeyError(
                "Resume audit key unavailable: "
                f"checkpoint_key={checkpoint_key!r}, runtime_key={runtime_key!r}"
            )
        shard_path = checkpoint / weight_map[checkpoint_key]
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            saved = handle.get_tensor(checkpoint_key)
        loaded = model_state[runtime_key].detach().cpu()
        exact_match = bool(torch.equal(saved, loaded))
        records.append(
            {
                "checkpoint_key": checkpoint_key,
                "runtime_key": runtime_key,
                "shape": list(saved.shape),
                "dtype": str(saved.dtype),
                "checkpoint_sha256": _tensor_sha256(saved),
                "runtime_sha256": _tensor_sha256(loaded),
                "exact_match": exact_match,
            }
        )
        if not exact_match:
            raise RuntimeError(
                "Conversion-aware resume audit failed for "
                f"{checkpoint_key!r} -> {runtime_key!r}"
            )
    audit_dir = output_dir / "resume_load_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint),
        "loader": "Qwen2_5_VLForConditionalGeneration.from_pretrained",
        "conversion_mapping": dict(model._checkpoint_conversion_mapping),
        "all_exact_match": True,
        "records": records,
    }
    (audit_dir / f"{checkpoint.name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _resolve_task_args(parse_args())
    world_size, global_rank, _ = _distributed_context()
    is_main_process = global_rank == 0
    if args.bf16 and args.fp16:
        raise SystemExit("Choose at most one of --bf16 and --fp16")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pixel_budget = args.image_size * args.image_size
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
        min_pixels=pixel_budget,
        max_pixels=pixel_budget,
    )
    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    model_load_source = args.resume_from_checkpoint or args.model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_load_source,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    top_level_model_type = model.config.to_dict().get("model_type")
    if top_level_model_type != "qwen2_5_vl":
        raise ValueError(
            "This direct-SFT control is Qwen2.5-VL-only; "
            f"got top-level model_type={top_level_model_type!r}"
        )
    embedding_rows = int(model.get_input_embeddings().weight.shape[0])
    if embedding_rows < len(processor.tokenizer):
        raise ValueError(
            "Stock Qwen checkpoint has fewer embedding rows than tokenizer entries: "
            f"embeddings={embedding_rows}, tokenizer={len(processor.tokenizer)}"
        )
    for parameter in model.visual.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    if is_main_process and args.resume_from_checkpoint is not None:
        _audit_conversion_aware_resume(model, args.resume_from_checkpoint, output_dir)

    dataset = WMVLMManifestDataset(
        args.manifest,
        root_dir=args.data_root,
        expected_examples=args.expected_train_examples,
        minimum_input_images=args.minimum_input_images,
        exact_input_images=args.exact_input_images,
        expected_source_format=args.expected_source_format,
        task_name=args.task,
    )
    if args.supervision_mode == "interleaved_image_placeholder":
        collator = QwenInterleavedImagePlaceholderSFTDataCollator(processor)
    elif args.supervision_mode == "reasoning_imagination_with_helper_image":
        collator = QwenImaginationWithHelperImageSFTDataCollator(processor)
    elif args.supervision_mode == "reasoning_imagination":
        collator = QwenImaginationSFTDataCollator(processor)
    else:
        collator = QwenDirectSFTDataCollator(processor)
    if is_main_process:
        _write_run_config(output_dir, args, dataset, world_size=world_size)
        processor.save_pretrained(output_dir)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if is_main_process:
        print(
            f"[qwen-sft] task={args.task} supervision={args.supervision_mode} "
            f"examples={len(dataset)} "
            f"total_rows={dataset.total_rows} "
            f"short_rows={dataset.skipped_short_rows} trainable={trainable:,}/{total:,} "
            f"world_size={world_size} effective_batch="
            f"{world_size * args.gradient_accumulation_steps}",
            flush=True,
        )

    report_to = [] if args.report_to.lower() in {"", "none"} else [args.report_to]
    training_args = TrainingArguments(
        output_dir=os.fspath(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="adamw_torch_fused",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=report_to,
        run_name=args.run_name,
        seed=args.seed,
        data_seed=args.seed,
        save_safetensors=True,
    )
    trainer = QwenConversionAwareResumeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
        preloaded_resume_checkpoint=args.resume_from_checkpoint,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if not args.skip_final_save:
        trainer.save_model(output_dir)
        if trainer.is_world_process_zero():
            processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
