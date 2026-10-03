#!/usr/bin/env python3
"""Evaluate stock Qwen2.5-VL controls on Tetris splits."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_str = os.fspath(REPO_ROOT)
while repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)

from evaluate_vlm.qwen25_vl_hf_backend import Qwen25VLHFBackend
from wm_vlm.data import IMAGINATION_PLACEHOLDER, build_imagination_user_prompt


DEFAULT_ID_DATA_ROOT = REPO_ROOT / "datasets/Tetris-2D-ID"
DEFAULT_OOD_DATA_ROOT = REPO_ROOT / "datasets/Tetris-2D-OOD"
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
    parser.add_argument(
        "--eval-json",
        type=Path,
        default=DEFAULT_ID_DATA_ROOT / "eval.json",
    )
    parser.add_argument(
        "--held-out-json",
        type=Path,
        default=DEFAULT_OOD_DATA_ROOT / "held_out.json",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["id_eval", "ood"],
        default=["id_eval", "ood"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--prompt-mode",
        choices=["answer_only", "reasoning_imagination"],
        default="answer_only",
    )
    parser.add_argument("--log-every", type=int, default=20)
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


def _resolve_image(json_path: Path, image_path: str) -> Path:
    candidate = Path(image_path)
    return candidate if candidate.is_absolute() else json_path.parent / candidate


def _load_split(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return rows if limit is None else rows[:limit]


def build_evaluation_prompt(question: str, prompt_mode: str) -> str:
    if prompt_mode == "answer_only":
        return question.strip() + "\nAnswer with the option letter only."
    if prompt_mode == "reasoning_imagination":
        return build_imagination_user_prompt(question)
    raise ValueError(f"Unsupported prompt_mode={prompt_mode!r}")


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
    prompt_mode: str,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    gold_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    output_path = output_dir / f"{name}_predictions.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            image_path = _resolve_image(json_path, str(row["img_path"]))
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing {name} image at row {index}: {image_path}"
                )
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
            prompt = build_evaluation_prompt(str(row["question"]), prompt_mode)
            prediction_text = backend.generate([prompt], [[rgb]])[0]
            predicted = extract_answer(prediction_text)
            gold = str(row["answer"]).strip().upper()
            if gold not in {"A", "B", "C", "D"}:
                raise ValueError(
                    f"Invalid gold answer at {name} row {index}: {gold!r}"
                )
            result = {
                "split": name,
                "index": index,
                "sample_id": row.get("sample_id"),
                "gold": gold,
                "predicted": predicted,
                "correct": predicted == gold,
                "prediction": prediction_text,
                "imagination_placeholder_count": prediction_text.count(
                    IMAGINATION_PLACEHOLDER
                ),
                "prompt": prompt,
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
                (index + 1) % log_every == 0 or index + 1 == len(rows)
            ):
                correct = sum(item["correct"] for item in predictions)
                print(
                    f"[eval] split={name} progress={index + 1}/{len(rows)} "
                    f"accuracy={correct / len(predictions):.4f}",
                    flush=True,
                )

    total = len(predictions)
    correct = sum(row["correct"] for row in predictions)
    parse_failures = sum(row["predicted"] is None for row in predictions)
    placeholder_examples = sum(
        row["imagination_placeholder_count"] > 0 for row in predictions
    )
    placeholder_count = sum(
        row["imagination_placeholder_count"] for row in predictions
    )
    return {
        "split": name,
        "inference_condition": (
            "free_reasoning_with_imagination_placeholder"
            if prompt_mode == "reasoning_imagination"
            else "standard_inference"
        ),
        "dataset_path": os.fspath(json_path),
        "predictions_path": os.fspath(output_path),
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "parse_failures": parse_failures,
        "parse_success_rate": (total - parse_failures) / total,
        "imagination_placeholder_examples": placeholder_examples,
        "imagination_placeholder_rate": placeholder_examples / total,
        "imagination_placeholder_count": placeholder_count,
        "mean_imagination_placeholder_count": placeholder_count / total,
        "gold_answers": dict(sorted(gold_counts.items())),
        "predicted_answers": dict(sorted(predicted_counts.items())),
        "by_transform": _summarize_grouped(predictions, "transform_description"),
        "by_shape_C_family": _summarize_grouped(predictions, "shape_C_family"),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
    reasoning_imagination = args.prompt_mode == "reasoning_imagination"
    metrics: dict[str, Any] = {
        "method": (
            "ordinary_qwen_reasoning_imagination_sft"
            if reasoning_imagination
            else "ordinary_qwen_direct_answer_sft"
        ),
        "model": args.model,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "image_size": args.image_size,
        "inference_condition": (
            "free_reasoning_with_imagination_placeholder"
            if reasoning_imagination
            else "standard_inference"
        ),
        "prompt_mode": args.prompt_mode,
        "uses_intermediate_reasoning_text": reasoning_imagination,
        "uses_intermediate_reasoning_image": False,
        "intermediate_image_replacement": (
            IMAGINATION_PLACEHOLDER if reasoning_imagination else None
        ),
        "uses_wm_vlm_latents": False,
        "splits": {},
    }
    for split in args.splits:
        path = split_paths[split]
        rows = _load_split(path, args.limit)
        metrics["splits"][split] = evaluate_split(
            name=split,
            json_path=path,
            rows=rows,
            backend=backend,
            output_dir=output_dir,
            log_every=args.log_every,
            prompt_mode=args.prompt_mode,
        )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"[eval] metrics={metrics_path}", flush=True)


if __name__ == "__main__":
    main()
