#!/usr/bin/env python3
"""Matched latent retrieval and causal latent-injection evaluation on Tetris."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

from PIL import Image
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_str = os.fspath(REPO_ROOT)
while repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)

from evaluate_vlm.wm_vlm_backend import WMVLMBackend
from wm_vlm.inference import encode_oracle_image_latents, generate_wm_vlm_answer


LETTERS = ("A", "B", "C", "D")
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
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def extract_answer(text: str) -> str | None:
    for pattern in (ANSWER_TAG_RE, ANSWER_TEXT_RE):
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()
    matches = OPTION_RE.findall(text)
    if matches:
        return matches[-1].upper()
    matches = BARE_RE.findall(text)
    return matches[-1].upper() if matches else None


def resolve_path(json_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else json_path.parent / path


def state_paths(json_path: Path, row: dict[str, Any]) -> list[Path]:
    trace = row.get("reasoning_trace")
    if not isinstance(trace, dict) or not isinstance(trace.get("items"), list):
        raise ValueError(f"{row.get('sample_id')}: missing reasoning_trace.items")
    paths = [
        resolve_path(json_path, str(item["file"]))
        for item in trace["items"]
        if isinstance(item, dict) and item.get("type") == "state"
    ]
    if len(paths) not in {1, 2}:
        raise ValueError(
            f"{row.get('sample_id')}: expected one or two state images, got {len(paths)}"
        )
    return paths


def option_paths(json_path: Path, row: dict[str, Any]) -> list[Path]:
    options = row.get("options")
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError(f"{row.get('sample_id')}: expected four options")
    by_letter = {str(item["letter"]).upper(): item for item in options}
    if set(by_letter) != set(LETTERS):
        raise ValueError(f"{row.get('sample_id')}: invalid option letters")
    return [resolve_path(json_path, str(by_letter[letter]["file"])) for letter in LETTERS]


def load_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.convert("RGB")


@torch.inference_mode()
def encode_image(backend: WMVLMBackend, path: Path) -> torch.Tensor:
    return encode_oracle_image_latents(
        backend.model,
        backend.processor,
        load_rgb(path),
        image_processor=backend.helper_image_processor,
    )[0]


def finite(value: float) -> float | str:
    return value if math.isfinite(value) else ""


@torch.inference_mode()
def pair_metrics(source: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    source_float = source.float()
    target_float = target.float()
    difference = source_float - target_float
    target_rms = target_float.square().mean().sqrt().clamp_min(1e-12)
    return {
        "flat_cosine": float(
            F.cosine_similarity(source_float.flatten(), target_float.flatten(), dim=0)
        ),
        "token_cosine": float(
            F.cosine_similarity(source_float, target_float, dim=-1).mean()
        ),
        "mean_pool_cosine": float(
            F.cosine_similarity(
                source_float.mean(dim=0), target_float.mean(dim=0), dim=0
            )
        ),
        "mse": float(difference.square().mean()),
        "relative_rmse": float(difference.square().mean().sqrt() / target_rms),
        "source_rms": float(source_float.square().mean().sqrt()),
        "target_rms": float(target_rms),
    }


@torch.inference_mode()
def option_scores(latent: torch.Tensor, options: torch.Tensor) -> dict[str, Any]:
    latent_float = latent.float()
    options_float = options.float()
    flat = F.cosine_similarity(
        options_float.flatten(start_dim=1),
        latent_float.flatten().unsqueeze(0),
        dim=1,
    )
    token = F.cosine_similarity(
        options_float,
        latent_float.unsqueeze(0),
        dim=-1,
    ).mean(dim=1)
    pooled = F.cosine_similarity(
        options_float.mean(dim=1),
        latent_float.mean(dim=0).unsqueeze(0),
        dim=1,
    )
    top = int(flat.argmax())
    order = torch.argsort(flat, descending=True).tolist()
    return {
        "latent_top1": LETTERS[top],
        "latent_top1_margin": float(flat[order[0]] - flat[order[1]]),
        **{f"flat_cosine_{letter}": float(flat[index]) for index, letter in enumerate(LETTERS)},
        **{f"token_cosine_{letter}": float(token[index]) for index, letter in enumerate(LETTERS)},
        **{f"pooled_cosine_{letter}": float(pooled[index]) for index, letter in enumerate(LETTERS)},
    }


def empty_option_scores() -> dict[str, str]:
    return {
        "latent_top1": "",
        "latent_top1_margin": "",
        **{
            f"{metric}_{letter}": ""
            for metric in ("flat_cosine", "token_cosine", "pooled_cosine")
            for letter in LETTERS
        },
    }


def generation_kwargs(backend: WMVLMBackend, question: str, image: Image.Image) -> dict[str, Any]:
    return {
        "model": backend.model,
        "processor": backend.processor,
        "question": question,
        "images": [image],
        "max_new_tokens": backend.max_tokens,
        "flow_steps": backend.flow_steps,
        "flow_solver": backend.flow_solver,
        "flow_seed": backend.flow_seed,
        "flow_noise_device": backend.flow_noise_device,
        "flow_state_dtype": backend.flow_state_dtype,
        "temperature": backend.temperature,
        "top_p": backend.top_p,
        "visual_consumer_mode": backend.visual_consumer_mode,
    }


def normalized_random_like(target: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(target.shape, generator=generator, dtype=torch.float32)
    target_float = target.detach().float().cpu()
    noise = (noise - noise.mean()) / noise.std().clamp_min(1e-12)
    noise = noise * target_float.std() + target_float.mean()
    return noise.to(device=target.device, dtype=target.dtype)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate_split(
    *,
    split: str,
    json_path: Path,
    backend: WMVLMBackend,
    output_dir: Path,
    num_shards: int,
    shard_index: int,
    log_every: int,
) -> None:
    all_rows = json.loads(json_path.read_text(encoding="utf-8"))
    selected = [
        (index, row)
        for index, row in enumerate(all_rows)
        if index % num_shards == shard_index
    ]
    condition_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    condition_path = output_dir / f"{split}_conditions.csv"
    block_path = output_dir / f"{split}_blocks.csv"

    for local_index, (global_index, row) in enumerate(selected):
        sample_id = str(row["sample_id"])
        gold = str(row.get("answer_letter", row.get("answer"))).strip().upper()
        if gold not in LETTERS:
            raise ValueError(f"{sample_id}: invalid gold answer {gold!r}")
        question = str(row["question"])
        problem_path = resolve_path(json_path, str(row["img_path"]))
        problem = load_rgb(problem_path)
        gt_paths = state_paths(json_path, row)
        gt_blocks = torch.stack([encode_image(backend, path) for path in gt_paths])
        options = torch.stack(
            [encode_image(backend, path) for path in option_paths(json_path, row)]
        )

        standard_diagnostics: dict[str, Any] = {}
        standard_text = generate_wm_vlm_answer(
            **generation_kwargs(backend, question, problem),
            capture_generated_latents=True,
            diagnostics=standard_diagnostics,
        )
        raw_generated = standard_diagnostics.pop("generated_latent_blocks")
        if not isinstance(raw_generated, list):
            raise TypeError("generated_latent_blocks must be a list")
        generated = [
            block[0].to(device=gt_blocks.device, dtype=gt_blocks.dtype)
            for block in raw_generated
        ]
        expected_blocks = int(gt_blocks.shape[0])
        emitted_blocks = len(generated)
        for block_index in range(expected_blocks):
            metrics: dict[str, float] | None = None
            if block_index < emitted_blocks:
                metrics = pair_metrics(generated[block_index], gt_blocks[block_index])
            block_rows.append(
                {
                    "split": split,
                    "index": global_index,
                    "sample_id": sample_id,
                    "block_index": block_index + 1,
                    "expected_blocks": expected_blocks,
                    "emitted_blocks": emitted_blocks,
                    "generated_block_present": block_index < emitted_blocks,
                    **{
                        key: finite(metrics[key]) if metrics is not None else ""
                        for key in (
                            "flat_cosine",
                            "token_cosine",
                            "mean_pool_cosine",
                            "mse",
                            "relative_rmse",
                            "source_rms",
                            "target_rms",
                        )
                    },
                }
            )

        wrong_index = (LETTERS.index(gold) + 1) % len(LETTERS)
        wrong_letter = LETTERS[wrong_index]
        donor_index = (global_index + 1) % len(all_rows)
        donor_row = all_rows[donor_index]
        donor_final_path = state_paths(json_path, donor_row)[-1]
        donor_final = encode_image(backend, donor_final_path)
        random_final = normalized_random_like(gt_blocks[-1], 20_000_000 + global_index)

        conditions: list[tuple[str, torch.Tensor | None, str | None, str | None]] = [
            ("standard_generated", None, None, None),
            ("oracle_correct", gt_blocks, gold, None),
        ]
        wrong_blocks = gt_blocks.clone()
        wrong_blocks[-1] = options[wrong_index]
        conditions.append(("oracle_wrong_option", wrong_blocks, wrong_letter, None))
        shuffled_blocks = gt_blocks.clone()
        shuffled_blocks[-1] = donor_final
        conditions.append(
            (
                "oracle_shuffled_sample",
                shuffled_blocks,
                None,
                str(donor_row["sample_id"]),
            )
        )
        random_blocks = gt_blocks.clone()
        random_blocks[-1] = random_final
        conditions.append(("oracle_random_matched_stats", random_blocks, None, None))

        for condition, injected, intended_option, donor_sample_id in conditions:
            if condition == "standard_generated":
                prediction_text = standard_text
                diagnostics = standard_diagnostics
                final_latent = generated[-1] if generated else None
                condition_emitted = emitted_blocks
            else:
                assert injected is not None
                diagnostics = {}
                prediction_text = generate_wm_vlm_answer(
                    **generation_kwargs(backend, question, problem),
                    oracle_latents=injected,
                    max_latent_blocks=expected_blocks,
                    diagnostics=diagnostics,
                )
                final_latent = injected[-1]
                condition_emitted = int(diagnostics["latent_blocks"])
            predicted = extract_answer(prediction_text)
            scores = (
                option_scores(final_latent, options)
                if final_latent is not None
                else empty_option_scores()
            )
            latent_top1 = scores.get("latent_top1")
            gold_score = scores.get(f"flat_cosine_{gold}")
            wrong_scores = [
                scores.get(f"flat_cosine_{letter}")
                for letter in LETTERS
                if letter != gold
            ]
            condition_rows.append(
                {
                    "split": split,
                    "index": global_index,
                    "sample_id": sample_id,
                    "condition": condition,
                    "gold": gold,
                    "predicted": predicted or "",
                    "answer_correct": predicted == gold,
                    "parse_success": predicted is not None,
                    "expected_blocks": expected_blocks,
                    "emitted_blocks": condition_emitted,
                    "block_count_match": condition_emitted == expected_blocks,
                    "latent_present": final_latent is not None,
                    "latent_top1": latent_top1 or "",
                    "latent_top1_is_gold": latent_top1 == gold,
                    "answer_matches_latent_top1": predicted is not None and predicted == latent_top1,
                    "intended_option": intended_option or "",
                    "answer_matches_intended_option": (
                        "" if intended_option is None else predicted == intended_option
                    ),
                    "donor_sample_id": donor_sample_id or "",
                    "gold_flat_cosine": finite(float(gold_score)) if gold_score is not None else "",
                    "gold_vs_best_wrong_margin": (
                        finite(float(gold_score - max(wrong_scores)))
                        if gold_score is not None and all(value is not None for value in wrong_scores)
                        else ""
                    ),
                    **{
                        key: "" if value == "" else finite(float(value))
                        for key, value in scores.items()
                        if key != "latent_top1"
                    },
                    "prediction_text": prediction_text.replace("\n", "\\n"),
                }
            )

        condition_fields = list(condition_rows[0])
        block_fields = list(block_rows[0])
        write_csv(condition_path, condition_rows, condition_fields)
        write_csv(block_path, block_rows, block_fields)
        if log_every > 0 and (
            (local_index + 1) % log_every == 0 or local_index + 1 == len(selected)
        ):
            standard_correct = sum(
                item["answer_correct"]
                for item in condition_rows
                if item["condition"] == "standard_generated"
            )
            print(
                f"[matched] split={split} shard={shard_index} "
                f"progress={local_index + 1}/{len(selected)} "
                f"standard_accuracy={standard_correct / (local_index + 1):.4f}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backend = WMVLMBackend(
        args.model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        max_latent_steps=1,
        flow_solver="euler",
        flow_seed=0,
        flow_noise_device="cpu",
        flow_state_dtype="bfloat16",
        top_p=1.0,
        dtype="bfloat16",
        device=args.device,
        attn_implementation="sdpa",
    )
    for split, json_path in (
        ("C-ID", args.eval_json),
        ("OOD", args.held_out_json),
    ):
        evaluate_split(
            split=split,
            json_path=json_path,
            backend=backend,
            output_dir=args.output_dir,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            log_every=args.log_every,
        )


if __name__ == "__main__":
    main()
