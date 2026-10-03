#!/usr/bin/env python3
"""Evaluate ordinary Qwen image-placeholder SFT with its exact training prompt."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root = os.fspath(REPO_ROOT)
while repo_root in sys.path:
    sys.path.remove(repo_root)
sys.path.insert(0, repo_root)

from evaluate_vlm.qwen25_vl_hf_backend import Qwen25VLHFBackend  # noqa: E402
from wm_vlm.constants import WM_VLM_SYSTEM_PROMPT  # noqa: E402
from wm_vlm.data import IMAGINATION_PLACEHOLDER  # noqa: E402


ANSWER_TAG_RE = re.compile(r"<answer>\s*([a-d])\s*</answer>", re.IGNORECASE)
ANSWER_TEXT_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:)?\s*(?:option\s*)?\(?([a-d])\)?",
    re.IGNORECASE,
)
OPTION_RE = re.compile(r"option\s*\(([a-d])\)", re.IGNORECASE)
BARE_RE = re.compile(r"\b([a-d])\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--held-out-json", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Resolve relative image paths against this directory instead of the "
            "directory containing each split JSON."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("id_eval", "ood"),
        default=("id_eval", "ood"),
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def extract_answer(text: str) -> str | None:
    for pattern in (ANSWER_TAG_RE, ANSWER_TEXT_RE):
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()
    option_matches = OPTION_RE.findall(text)
    if option_matches:
        return option_matches[-1].upper()
    bare_matches = BARE_RE.findall(text)
    return bare_matches[-1].upper() if bare_matches else None


def _resolve_image(
    json_path: Path,
    image_path: str,
    image_root: Path | None = None,
) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate
    return (json_path.parent if image_root is None else image_root) / candidate


def _load_shard(path: Path, num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(
            f"Invalid sharding: num_shards={num_shards}, shard_index={shard_index}"
        )
    selected = []
    for index, row in enumerate(rows):
        if index % num_shards == shard_index:
            copied = dict(row)
            copied["__evaluation_index"] = index
            selected.append(copied)
    return selected


@torch.inference_mode()
def generate_with_exact_training_prompt(
    backend: Qwen25VLHFBackend,
    question: str,
    image: Image.Image,
) -> str:
    messages = [
        {"role": "system", "content": WM_VLM_SYSTEM_PROMPT.strip()},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question.strip()},
            ],
        },
    ]
    rendered = backend.processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    batch = backend.processor(
        text=[rendered],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    device = next(backend.model.parameters()).device
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    generated = backend.model.generate(
        **batch,
        max_new_tokens=backend.max_tokens,
        do_sample=False,
        use_cache=True,
        temperature=None,
        top_p=None,
    )
    prompt_length = int(batch["input_ids"].shape[1])
    return backend.processor.tokenizer.decode(
        generated[0, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def _summarize_grouped(
    predictions: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get(key, "unknown"))].append(bool(row["correct"]))
    return {
        name: {
            "correct": sum(values),
            "total": len(values),
            "accuracy": sum(values) / len(values),
        }
        for name, values in sorted(grouped.items())
    }


def evaluate_split(
    *,
    name: str,
    json_path: Path,
    rows: list[dict[str, Any]],
    backend: Qwen25VLHFBackend,
    output_dir: Path,
    log_every: int,
    image_root: Path | None = None,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    gold_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    output_path = output_dir / f"{name}_predictions.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for local_index, row in enumerate(rows):
            evaluation_index = int(row["__evaluation_index"])
            image_path = _resolve_image(
                json_path,
                str(row["img_path"]),
                image_root,
            )
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing {name} image at index {evaluation_index}: {image_path}"
                )
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            prediction_text = generate_with_exact_training_prompt(
                backend,
                str(row["question"]),
                image,
            )
            predicted = extract_answer(prediction_text)
            gold = str(row["answer"]).strip().upper()
            if gold not in {"A", "B", "C", "D"}:
                raise ValueError(
                    f"Invalid gold answer at {name} index {evaluation_index}: {gold!r}"
                )
            result = {
                "split": name,
                "index": evaluation_index,
                "sample_id": row.get("sample_id"),
                "gold": gold,
                "predicted": predicted,
                "correct": predicted == gold,
                "prediction": prediction_text,
                "imagination_placeholder_count": prediction_text.count(
                    IMAGINATION_PLACEHOLDER
                ),
                "image_path": os.fspath(image_path),
                "transform_description": row.get("transform_description"),
                "shape_A_family": row.get("shape_A_family"),
                "shape_C_family": row.get("shape_C_family"),
                "shape_C_name": row.get("shape_C_name"),
            }
            predictions.append(result)
            gold_counts[gold] += 1
            predicted_counts[predicted or "<parse_failure>"] += 1
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if log_every > 0 and (
                (local_index + 1) % log_every == 0
                or local_index + 1 == len(rows)
            ):
                correct = sum(bool(item["correct"]) for item in predictions)
                print(
                    f"[eval] split={name} progress={local_index + 1}/{len(rows)} "
                    f"accuracy={correct / len(predictions):.4f}",
                    flush=True,
                )

    total = len(predictions)
    correct = sum(bool(row["correct"]) for row in predictions)
    parse_failures = sum(row["predicted"] is None for row in predictions)
    marker_examples = sum(
        int(row["imagination_placeholder_count"]) > 0 for row in predictions
    )
    marker_count = sum(
        int(row["imagination_placeholder_count"]) for row in predictions
    )
    return {
        "split": name,
        "inference_condition": "training_exact_free_reasoning",
        "dataset_path": os.fspath(json_path),
        "predictions_path": os.fspath(output_path),
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "parse_failures": parse_failures,
        "parse_success_rate": (total - parse_failures) / total,
        "imagination_placeholder_examples": marker_examples,
        "imagination_placeholder_rate": marker_examples / total,
        "imagination_placeholder_count": marker_count,
        "mean_imagination_placeholder_count": marker_count / total,
        "gold_answers": dict(sorted(gold_counts.items())),
        "predicted_answers": dict(sorted(predicted_counts.items())),
        "by_transform": _summarize_grouped(predictions, "transform_description"),
        "by_shape_C_family": _summarize_grouped(predictions, "shape_C_family"),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = args.image_root.resolve() if args.image_root is not None else None
    backend = Qwen25VLHFBackend(
        args.model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        top_p=1.0,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        image_size=args.image_size,
    )
    split_paths = {
        "id_eval": args.eval_json.resolve(),
        "ood": args.held_out_json.resolve(),
    }
    metrics: dict[str, Any] = {
        "method": "stock_qwen_ordinary_interleaved_image_placeholder_sft",
        "model": args.model,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "image_size": args.image_size,
        "inference_condition": "training_exact_free_reasoning",
        "prompt_mode": "ordinary_interleaved_image_placeholder_training_exact",
        "system_prompt": WM_VLM_SYSTEM_PROMPT,
        "adds_imagination_instruction": False,
        "uses_intermediate_reasoning_image": False,
        "intermediate_image_replacement": IMAGINATION_PLACEHOLDER,
        "uses_wm_vlm_latents": False,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "image_root": str(image_root) if image_root is not None else None,
        "splits": {},
    }
    for split in args.splits:
        path = split_paths[split]
        metrics["splits"][split] = evaluate_split(
            name=split,
            json_path=path,
            rows=_load_shard(path, args.num_shards, args.shard_index),
            backend=backend,
            output_dir=output_dir,
            log_every=args.log_every,
            image_root=image_root,
        )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[eval] metrics={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
