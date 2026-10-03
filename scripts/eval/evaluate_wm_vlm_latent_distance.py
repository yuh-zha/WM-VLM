#!/usr/bin/env python3
"""Measure generated WM-VLM visual latents against oracle vision latents."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root = os.fspath(REPO_ROOT)
while repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)

from evaluate_vlm.wm_vlm_backend import (  # noqa: E402
    WMVLMBackend,
)
from wm_vlm.generation import (  # noqa: E402
    _cached_problem_features,
    _stock_text_logits,
    integrate_flow_tokens,
)
from wm_vlm.metrics import (  # noqa: E402
    LATENT_METRIC_NAMES as METRIC_NAMES,
    latent_distance_metrics as latent_metrics,
)
from wm_vlm.data import (  # noqa: E402
    build_user_messages,
    preprocess_with_qwen_vl_utils,
)
from wm_vlm.generation import (  # noqa: E402
    _eos_ids,
    _sample_token,
)
from wm_vlm.inference import encode_oracle_image_latents  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--held-out-json", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("id_eval", "ood"),
        default=("id_eval", "ood"),
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--flow-solver", choices=("euler", "heun"), default="heun")
    parser.add_argument("--flow-seed", type=int, default=0)
    parser.add_argument("--max-prefix-tokens", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def _resolve_image(json_path: Path, image_path: str) -> Path:
    candidate = Path(image_path)
    return candidate if candidate.is_absolute() else json_path.parent / candidate


def _oracle_image_path(json_path: Path, row: dict[str, Any]) -> Path:
    traces = row.get("reasoning_traces")
    if not isinstance(traces, dict):
        raise ValueError("Latent evaluation requires reasoning_traces")
    path = traces.get("intermediate_img_path")
    if not isinstance(path, str) or not path:
        raise ValueError("Missing reasoning_traces.intermediate_img_path")
    return _resolve_image(json_path, path)


def _load_rows(path: Path, *, num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if index % num_shards != shard_index:
            continue
        copied = dict(row)
        copied["__evaluation_index"] = index
        rows.append(copied)
    return rows


@torch.inference_mode()
def sample_generated_latents(
    backend: WMVLMBackend,
    *,
    question: str,
    image: Image.Image,
    flow_steps: int,
    flow_solver: str,
    flow_seed: int,
    max_prefix_tokens: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    """Run the standard decoder until its visual block and return its ODE endpoint."""
    model = backend.model
    processor = backend.processor
    images: list[Any] = [image]
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
        images = [preprocess_with_qwen_vl_utils(item) for item in images]
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
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    image_grid_thw = batch.get("image_grid_thw")
    image_features = _cached_problem_features(model, batch)
    model.reset_rope_deltas()

    latent_id = int(model.config.wm_vlm_latent_token_id)
    latent_start_id = int(model.config.wm_vlm_latent_start_id)
    latent_size = int(model.config.wm_vlm_latent_size)
    hidden_size = int(model.config.text_config.hidden_size)
    eos_ids = _eos_ids(processor.tokenizer)
    generated_prefix: list[int] = []

    for prefix_step in range(max_prefix_tokens):
        logits = _stock_text_logits(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            image_features=image_features,
        )
        emitted_id = _sample_token(logits[:, -1, :], temperature=0.0, top_p=1.0)
        generated_prefix.append(emitted_id)
        new_id = torch.tensor([[emitted_id]], dtype=input_ids.dtype, device=device)
        input_ids = torch.cat((input_ids, new_id), dim=1)
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones((1, 1), dtype=attention_mask.dtype, device=device),
            ),
            dim=1,
        )
        if emitted_id == latent_start_id:
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
            generator = torch.Generator(device=device)
            generator.manual_seed(flow_seed)
            initial = torch.randn(
                (1, latent_size, hidden_size),
                generator=generator,
                device=device,
                dtype=torch.float32,
            ) * float(model.config.wm_vlm_flow_noise_scale)

            def velocity(states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
                return model.predict_flow_velocity(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    flow_states=states,
                    flow_timesteps=times,
                    image_grid_thw=image_grid_thw,
                    image_features=image_features,
                )

            endpoint, flow_nfe = integrate_flow_tokens(
                velocity,
                initial,
                num_steps=flow_steps,
                solver=flow_solver,
            )
            return endpoint[0], initial[0], {
                "latent_generated": True,
                "prefix_steps": prefix_step + 1,
                "flow_nfe": flow_nfe,
                "generated_prefix": processor.tokenizer.decode(
                    generated_prefix,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            }
        if emitted_id in eos_ids:
            break
        decoded = processor.tokenizer.decode(
            generated_prefix,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if "</answer>" in decoded:
            break
    return None, None, {
        "latent_generated": False,
        "prefix_steps": len(generated_prefix),
        "flow_nfe": 0,
        "generated_prefix": processor.tokenizer.decode(
            generated_prefix,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "total": len(rows),
        "latent_generated": sum(bool(row["latent_generated"]) for row in rows),
    }
    output["latent_generation_rate"] = output["latent_generated"] / output["total"]
    comparisons: dict[str, Any] = {}
    for comparison in (
        "generated_vs_gt",
        "initial_noise_vs_gt",
        "generated_vs_shuffled_gt",
    ):
        selected = [row[comparison] for row in rows if comparison in row]
        comparisons[comparison] = {
            "count": len(selected),
            "mean": {
                metric: sum(values[metric] for values in selected) / len(selected)
                for metric in METRIC_NAMES
            }
            if selected
            else {},
        }
    output["comparisons"] = comparisons
    return output


def evaluate_split(
    backend: WMVLMBackend,
    *,
    split: str,
    json_path: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    first_target: torch.Tensor | None = None
    previous_prediction: torch.Tensor | None = None
    previous_entry_index: int | None = None

    for local_index, row in enumerate(rows):
        problem_path = _resolve_image(json_path, str(row["img_path"]))
        oracle_path = _oracle_image_path(json_path, row)
        if not problem_path.is_file() or not oracle_path.is_file():
            raise FileNotFoundError(
                f"Missing images for {split} row {row['__evaluation_index']}: "
                f"problem={problem_path}, oracle={oracle_path}"
            )
        with Image.open(problem_path) as handle:
            problem_image = handle.convert("RGB")
        with Image.open(oracle_path) as handle:
            oracle_image = handle.convert("RGB")
        target = encode_oracle_image_latents(
            backend.model,
            backend.processor,
            oracle_image,
            image_processor=backend.helper_image_processor,
        )[0].float()
        prediction, initial, diagnostics = sample_generated_latents(
            backend,
            question=str(row["question"]),
            image=problem_image,
            flow_steps=args.flow_steps,
            flow_solver=args.flow_solver,
            flow_seed=args.flow_seed,
            max_prefix_tokens=args.max_prefix_tokens,
        )
        entry: dict[str, Any] = {
            "split": split,
            "index": int(row["__evaluation_index"]),
            "sample_id": row.get("sample_id"),
            **diagnostics,
        }
        if prediction is not None and initial is not None:
            entry["generated_vs_gt"] = latent_metrics(prediction, target)
            entry["initial_noise_vs_gt"] = latent_metrics(initial, target)
            if first_target is None:
                first_target = target.detach().clone()
            if previous_prediction is not None and previous_entry_index is not None:
                entries[previous_entry_index]["generated_vs_shuffled_gt"] = latent_metrics(
                    previous_prediction,
                    target,
                )
            previous_prediction = prediction.detach().clone()
            previous_entry_index = len(entries)
        entries.append(entry)
        if args.log_every > 0 and (
            (local_index + 1) % args.log_every == 0 or local_index + 1 == len(rows)
        ):
            latest = entry.get("generated_vs_gt", {})
            print(
                f"[latent-distance] split={split} shard={args.shard_index}/"
                f"{args.num_shards} progress={local_index + 1}/{len(rows)} "
                f"generated={diagnostics['latent_generated']} "
                f"prefix_steps={diagnostics['prefix_steps']} "
                f"mse={latest.get('mse')} cosine={latest.get('cosine')}",
                flush=True,
            )

    if (
        previous_prediction is not None
        and previous_entry_index is not None
        and first_target is not None
    ):
        entries[previous_entry_index]["generated_vs_shuffled_gt"] = latent_metrics(
            previous_prediction,
            first_target,
        )
    return entries, _summary(entries)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be smaller than --num-shards")
    if args.flow_steps < 1 or args.max_prefix_tokens < 1:
        raise ValueError("Flow and prefix step counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = WMVLMBackend(
        args.model,
        max_tokens=args.max_prefix_tokens,
        max_latent_steps=args.flow_steps,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )
    split_paths = {
        "id_eval": args.eval_json.resolve(),
        "ood": args.held_out_json.resolve(),
    }
    all_entries: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for split in args.splits:
        rows = _load_rows(
            split_paths[split],
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        entries, summary = evaluate_split(
            backend,
            split=split,
            json_path=split_paths[split],
            rows=rows,
            args=args,
        )
        all_entries.extend(entries)
        summaries[split] = summary

    examples_path = args.output_dir / "per_example.jsonl"
    with examples_path.open("w", encoding="utf-8") as handle:
        for entry in all_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    payload = {
        "model": args.model,
        "variant": backend.model.config.wm_vlm_variant,
        "routing_mode": backend.routing_mode,
        "layer_placement": backend.layer_placement,
        "qkv_routing": backend.model.config.wm_vlm_qkv_routing,
        "latent_size": backend.latent_size,
        "hidden_size": int(backend.model.config.text_config.hidden_size),
        "flow_steps": args.flow_steps,
        "flow_solver": args.flow_solver,
        "flow_seed": args.flow_seed,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "per_example": os.fspath(examples_path),
        "splits": summaries,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print(f"[latent-distance] metrics={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
