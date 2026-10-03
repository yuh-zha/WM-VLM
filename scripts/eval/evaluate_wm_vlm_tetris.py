#!/usr/bin/env python3
"""Evaluate a WM-VLM checkpoint on Tetris ID and unseen-shape splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import inspect
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_str = os.fspath(REPO_ROOT)
# When invoked as ``python scripts/eval/<script>.py``, Python puts
# ``scripts/eval`` ahead of PYTHONPATH.  That directory also contains
# ``evaluate_vlm.py``, which would shadow the repository-level package.
while repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)

from evaluate_vlm.wm_vlm_backend import WMVLMBackend


DEFAULT_ID_DATA_ROOT = REPO_ROOT / "datasets/Tetris-2D-ID"
DEFAULT_OOD_DATA_ROOT = REPO_ROOT / "datasets/Tetris-2D-OOD"
GT_INTERMEDIATE_PROMPT_TEMPLATE = (
    "The first image is the original problem image. The second image is the "
    "ground-truth intermediate reasoning image showing the result of applying "
    "the required transformation to the query shape. Use both images to answer "
    "the original multiple-choice question. Do not treat the second image as an "
    "answer option.\n\nOriginal question:\n{question}"
)
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
        "--parquet-split",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Read an evaluation split directly from an image-bearing Parquet "
            "file with Hugging Face datasets. Repeat for multiple splits."
        ),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help=(
            "Resolve relative image paths against this directory instead of the "
            "directory containing each split JSON."
        ),
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-latent-steps", type=int, default=8)
    if "flow_solver" in inspect.signature(WMVLMBackend).parameters:
        parser.add_argument("--flow-solver", choices=("euler", "heun"), default="heun")
    if "flow_seed" in inspect.signature(WMVLMBackend).parameters:
        parser.add_argument("--flow-seed", type=int, default=0)
    if "flow_noise_device" in inspect.signature(WMVLMBackend).parameters:
        parser.add_argument(
            "--flow-noise-device",
            choices=("cpu", "model"),
            default="model",
        )
    if "flow_state_dtype" in inspect.signature(WMVLMBackend).parameters:
        parser.add_argument(
            "--flow-state-dtype",
            choices=("bfloat16", "float32"),
            default="float32",
        )
    if "save_generated_pixels_dir" in inspect.signature(WMVLMBackend).parameters:
        parser.add_argument(
            "--save-generated-pixels",
            action="store_true",
            help=(
                "Save each generated RGB reconstruction before it is passed "
                "through the frozen vision encoder."
            ),
        )
    if "visual_consumer_mode" in inspect.signature(WMVLMBackend).parameters:
        parser.add_argument(
            "--visual-consumer-mode",
            choices=(
                "checkpoint",
                "generated_pixel",
                "white_pixel",
                "shuffle_pixel_patches",
                "generated_endpoint",
            ),
            default="checkpoint",
            help="Override or ablate the completed visual state consumed by the VLM.",
        )
    if "generated_latent_ablation" in inspect.signature(WMVLMBackend).parameters:
        parser.add_argument(
            "--generated-latent-ablation",
            choices=("none", "zero", "random_matched_stats", "shuffle_tokens"),
            default="none",
            help="Transform each generated latent block before the VLM consumes it.",
        )
    if "generated_latent_ablation_seed" in inspect.signature(
        WMVLMBackend
    ).parameters:
        parser.add_argument(
            "--generated-latent-ablation-seed",
            type=int,
            default=0,
            help="Base CPU RNG seed for random or token-shuffle latent ablations.",
        )
    if hasattr(WMVLMBackend, "generate_with_flow_source"):
        parser.add_argument(
            "--flow-source-id-dir",
            type=Path,
            default=None,
            help="Directory containing ID query-shape crops matched by image basename.",
        )
        parser.add_argument(
            "--flow-source-ood-dir",
            type=Path,
            default=None,
            help="Directory containing OOD query-shape crops matched by image basename.",
        )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--oracle-latents",
        action="store_true",
        help=(
            "Replace generated continuous states with latents encoded from each "
            "sample's hidden intermediate image."
        ),
    )
    parser.add_argument(
        "--gt-intermediate-in-prompt",
        action="store_true",
        help=(
            "Expose the ground-truth intermediate reasoning image as a second "
            "ordinary prompt image. No continuous oracle latents are injected."
        ),
    )
    parser.add_argument(
        "--allow-stage1-latent-inference",
        action="store_true",
        help="Allow diagnostic generation from a Stage-1 checkpoint.",
    )
    if "force_oracle_blocks_at_start" in inspect.signature(WMVLMBackend).parameters:
        parser.add_argument(
            "--force-oracle-blocks-at-start",
            action="store_true",
            help=(
                "Force supplied oracle blocks into the assistant prefix before "
                "decoding; intended for Stage-1 checkpoints without a learned "
                "latent-sentinel policy."
            ),
        )
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


def _load_split(
    path: Path,
    limit: int | None,
    *,
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}); got {shard_index}"
        )
    limited = rows if limit is None else rows[:limit]
    if num_shards == 1:
        return limited
    sharded = []
    for original_index, row in enumerate(limited):
        if original_index % num_shards == shard_index:
            copied = dict(row)
            copied["__evaluation_index"] = original_index
            sharded.append(copied)
    return sharded


def _parse_parquet_splits(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(
                "--parquet-split must use NAME=PATH syntax; "
                f"got {value!r}"
            )
        name = name.strip()
        if name in paths:
            raise ValueError(f"Duplicate Parquet split name: {name!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing Parquet split {name!r}: {path}")
        paths[name] = path
    return paths


def _load_parquet_split(
    path: Path,
    limit: int | None,
    *,
    num_shards: int = 1,
    shard_index: int = 0,
    include_oracle_images: bool = False,
    include_flow_source_image: bool = False,
) -> Sequence[dict[str, Any]]:
    """Load embedded images without materializing them into standalone files."""
    try:
        from datasets import Image as DatasetImage
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "Parquet evaluation requires the Hugging Face datasets package"
        ) from error

    dataset = load_dataset(
        "parquet",
        data_files={"evaluation": os.fspath(path)},
        split="evaluation",
    )
    required = {"question", "answer", "composite_image"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Parquet split {path} is missing required columns: {sorted(missing)}"
        )
    image_columns = {
        name
        for name, feature in dataset.features.items()
        if isinstance(feature, DatasetImage)
    }
    retained_images = {"composite_image"}
    if include_oracle_images:
        retained_images.update(
            {
                "intermediate_rotation_image",
                "visual_cot_step_1_image",
                "visual_cot_step_2_image",
            }
        )
    if include_flow_source_image:
        retained_images.add("query_c_image")
    retained_columns = [
        name
        for name in dataset.column_names
        if name != "record_json"
        and (name not in image_columns or name in retained_images)
    ]
    dataset = dataset.select_columns(retained_columns)

    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}); got {shard_index}"
        )
    total = len(dataset) if limit is None else min(limit, len(dataset))
    indices = list(range(shard_index, total, num_shards))
    dataset = dataset.select(indices)
    return dataset.add_column("__evaluation_index", indices)


def _embedded_image_reference(
    source_path: Path,
    row: dict[str, Any],
    column: str,
) -> str:
    index = int(row.get("__evaluation_index", -1))
    return f"{source_path}#row={index}:{column}"


def _embedded_oracle_images(
    source_path: Path,
    row: dict[str, Any],
) -> tuple[list[Image.Image], list[str]]:
    columns = [
        name
        for name in ("visual_cot_step_1_image", "visual_cot_step_2_image")
        if row.get(name) is not None
    ]
    if not columns and row.get("intermediate_rotation_image") is not None:
        columns = ["intermediate_rotation_image"]
    images: list[Image.Image] = []
    references: list[str] = []
    for column in columns:
        image = row[column]
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected datasets.Image to decode {column!r} as PIL.Image, "
                f"got {type(image).__name__}"
            )
        images.append(image.convert("RGB"))
        references.append(_embedded_image_reference(source_path, row, column))
    return images, references


def _resolve_oracle_image(
    json_path: Path,
    row: dict[str, Any],
    image_root: Path | None = None,
) -> Path:
    traces = row.get("reasoning_traces")
    if not isinstance(traces, dict):
        raise ValueError("Oracle evaluation requires a reasoning_traces object")
    intermediate_path = traces.get("intermediate_img_path")
    if not isinstance(intermediate_path, str) or not intermediate_path:
        raise ValueError(
            "Oracle evaluation requires reasoning_traces.intermediate_img_path"
        )
    return _resolve_image(json_path, intermediate_path, image_root)


def _resolve_oracle_images(
    json_path: Path,
    row: dict[str, Any],
    image_root: Path | None = None,
) -> list[Path]:
    """Resolve the ordered state images used by multiblock visual-CoT training."""
    trace = row.get("reasoning_trace")
    if isinstance(trace, dict):
        items = trace.get("items")
        if isinstance(items, list):
            state_paths: list[Path] = []
            for item in items:
                if not isinstance(item, dict) or item.get("type") != "state":
                    continue
                state_file = item.get("file")
                if not isinstance(state_file, str) or not state_file:
                    raise ValueError(
                        "Oracle evaluation requires a file for every reasoning_trace "
                        "state item"
                    )
                state_paths.append(
                    _resolve_image(json_path, state_file, image_root)
                )
            if state_paths:
                return state_paths
    # Backward compatibility for the original single-image Tetris schema.
    return [_resolve_oracle_image(json_path, row, image_root)]


def build_gt_intermediate_prompt(question: str) -> str:
    """Label the original problem and visible GT-intermediate prompt images."""
    return GT_INTERMEDIATE_PROMPT_TEMPLATE.format(question=question)


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
    source_path: Path,
    rows: Sequence[dict[str, Any]],
    backend: WMVLMBackend,
    output_dir: Path,
    log_every: int,
    use_oracle_latents: bool,
    gt_intermediate_in_prompt: bool,
    image_root: Path | None = None,
    flow_source_image_dir: Path | None = None,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    answer_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    output_path = output_dir / f"{name}_predictions.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            embedded_image = row.get("composite_image")
            image_path: Path | None = None
            if embedded_image is not None:
                if not isinstance(embedded_image, Image.Image):
                    raise TypeError(
                        "Expected datasets.Image to decode composite_image as "
                        f"PIL.Image, got {type(embedded_image).__name__}"
                    )
                rgb = embedded_image.convert("RGB")
                input_image_reference = _embedded_image_reference(
                    source_path,
                    row,
                    "composite_image",
                )
            else:
                image_path = _resolve_image(
                    source_path,
                    str(row["img_path"]),
                    image_root,
                )
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Missing {name} image at row {index}: {image_path}"
                    )
                with Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                input_image_reference = os.fspath(image_path)
            flow_source_image_path: Path | None = None
            flow_source_image_reference: str | None = None
            flow_source_rgb: Image.Image | None = None
            if (
                not use_oracle_latents
                and bool(getattr(backend, "requires_flow_source_image", False))
            ):
                embedded_flow_source = row.get("query_c_image")
                if embedded_flow_source is not None:
                    if not isinstance(embedded_flow_source, Image.Image):
                        raise TypeError(
                            "Expected datasets.Image to decode query_c_image as "
                            f"PIL.Image, got {type(embedded_flow_source).__name__}"
                        )
                    flow_source_rgb = embedded_flow_source.convert("RGB")
                    flow_source_image_reference = _embedded_image_reference(
                        source_path,
                        row,
                        "query_c_image",
                    )
                else:
                    if flow_source_image_dir is None or image_path is None:
                        raise ValueError(
                            "Checkpoint requires an embedded query_c_image or a "
                            f"query-crop directory for split {name}"
                        )
                    flow_source_image_path = flow_source_image_dir / image_path.name
                    if not flow_source_image_path.is_file():
                        raise FileNotFoundError(
                            f"Missing {name} query-crop flow source at row {index}: "
                            f"{flow_source_image_path}"
                        )
                    with Image.open(flow_source_image_path) as flow_source_image:
                        flow_source_rgb = flow_source_image.convert("RGB")
                    flow_source_image_reference = os.fspath(flow_source_image_path)
            intermediate_image_reference: str | None = None
            intermediate_rgb: Image.Image | None = None
            oracle_image_references: list[str] = []
            oracle_rgbs: list[Image.Image] = []
            if use_oracle_latents:
                if embedded_image is not None:
                    oracle_rgbs, oracle_image_references = _embedded_oracle_images(
                        source_path,
                        row,
                    )
                    if not oracle_rgbs:
                        raise ValueError(
                            f"Parquet split {name} has no embedded oracle image at row {index}"
                        )
                else:
                    oracle_image_paths = _resolve_oracle_images(
                        source_path,
                        row,
                        image_root,
                    )
                    for oracle_image_path in oracle_image_paths:
                        if not oracle_image_path.is_file():
                            raise FileNotFoundError(
                                f"Missing {name} oracle image at row {index}: "
                                f"{oracle_image_path}"
                            )
                        with Image.open(oracle_image_path) as oracle_image:
                            oracle_rgbs.append(oracle_image.convert("RGB"))
                    oracle_image_references = [
                        os.fspath(path) for path in oracle_image_paths
                    ]
                intermediate_image_reference = oracle_image_references[-1]
            elif gt_intermediate_in_prompt:
                if embedded_image is not None:
                    embedded_oracles, embedded_references = _embedded_oracle_images(
                        source_path,
                        row,
                    )
                    if not embedded_oracles:
                        raise ValueError(
                            f"Parquet split {name} has no embedded intermediate image "
                            f"at row {index}"
                        )
                    intermediate_rgb = embedded_oracles[-1]
                    intermediate_image_reference = embedded_references[-1]
                else:
                    intermediate_image_path = _resolve_oracle_image(
                        source_path,
                        row,
                        image_root,
                    )
                    if not intermediate_image_path.is_file():
                        raise FileNotFoundError(
                            f"Missing {name} intermediate image at row {index}: "
                            f"{intermediate_image_path}"
                        )
                    with Image.open(intermediate_image_path) as intermediate_image:
                        intermediate_rgb = intermediate_image.convert("RGB")
                    intermediate_image_reference = os.fspath(intermediate_image_path)
            previous_activations = backend.latent_activation_count
            previous_steps = backend.total_latent_steps
            previous_oracle_injections = backend.oracle_injection_count
            previous_oracle_steps = backend.total_oracle_latent_steps
            previous_flow_source_injections = int(
                getattr(backend, "flow_source_injection_count", 0)
            )
            question = str(row["question"])
            if use_oracle_latents:
                assert oracle_rgbs
                prediction_text = backend.generate_with_oracle(
                    [question],
                    [[rgb]],
                    [oracle_rgbs],
                )[0]
            elif gt_intermediate_in_prompt:
                assert intermediate_rgb is not None
                source_prompt = build_gt_intermediate_prompt(question)
                source_prompt_images = [rgb, intermediate_rgb]
                if flow_source_rgb is not None:
                    prediction_text = backend.generate_with_flow_source(
                        [source_prompt],
                        [source_prompt_images],
                        [flow_source_rgb],
                    )[0]
                else:
                    prediction_text = backend.generate(
                        [source_prompt],
                        [source_prompt_images],
                    )[0]
            else:
                if flow_source_rgb is not None:
                    prediction_text = backend.generate_with_flow_source(
                        [question],
                        [[rgb]],
                        [flow_source_rgb],
                    )[0]
                else:
                    prediction_text = backend.generate([question], [[rgb]])[0]
            generated_pixel_batches = getattr(
                backend,
                "last_generated_pixel_records",
                [],
            )
            if len(generated_pixel_batches) not in {0, 1}:
                raise RuntimeError(
                    "Expected generated-pixel records for one evaluated prompt, "
                    f"got {len(generated_pixel_batches)} batches"
                )
            generated_pixel_records = (
                generated_pixel_batches[0] if generated_pixel_batches else []
            )
            if embedded_image is not None:
                _, ground_truth_oracle_references = _embedded_oracle_images(
                    source_path,
                    row,
                )
            else:
                ground_truth_oracle_references = [
                    os.fspath(path)
                    for path in _resolve_oracle_images(
                        source_path,
                        row,
                        image_root,
                    )
                ]
            predicted = extract_answer(prediction_text)
            gold = str(row["answer"]).strip().upper()
            if gold not in {"A", "B", "C", "D"}:
                raise ValueError(f"Invalid gold answer at {name} row {index}: {gold!r}")
            result = {
                "split": name,
                "index": int(row.get("__evaluation_index", index)),
                "sample_id": row.get("sample_id"),
                "input_image_path": input_image_reference,
                "ground_truth_intermediate_image_path": (
                    ground_truth_oracle_references[-1]
                    if ground_truth_oracle_references
                    else None
                ),
                "ground_truth_oracle_image_paths": ground_truth_oracle_references,
                "gold": gold,
                "predicted": predicted,
                "correct": predicted == gold,
                "prediction": prediction_text,
                "generated_pixel_images": generated_pixel_records,
                "latent_activated": backend.latent_activation_count > previous_activations,
                "latent_steps": backend.total_latent_steps - previous_steps,
                "oracle_injected": (
                    backend.oracle_injection_count > previous_oracle_injections
                ),
                "oracle_latent_steps": (
                    backend.total_oracle_latent_steps - previous_oracle_steps
                ),
                "flow_source_injected": int(
                    getattr(backend, "flow_source_injection_count", 0)
                )
                > previous_flow_source_injections,
                "flow_source_image_path": flow_source_image_reference,
                "oracle_image_path": (
                    intermediate_image_reference
                    if use_oracle_latents
                    else None
                ),
                "oracle_image_paths": oracle_image_references,
                "oracle_block_count": len(oracle_image_references),
                "gt_intermediate_in_prompt": gt_intermediate_in_prompt,
                "prompt_image_count": 2 if gt_intermediate_in_prompt else 1,
                "intermediate_prompt_image_path": (
                    intermediate_image_reference
                    if gt_intermediate_in_prompt
                    else None
                ),
                "transform_description": row.get("transform_description"),
                "shape_A_family": row.get("shape_A_family"),
                "shape_C_family": row.get("shape_C_family"),
                "shape_C_name": row.get("shape_C_name"),
            }
            predictions.append(result)
            answer_counts[gold] += 1
            predicted_counts[predicted or "<parse_failure>"] += 1
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if log_every > 0 and ((index + 1) % log_every == 0 or index + 1 == len(rows)):
                correct = sum(item["correct"] for item in predictions)
                print(
                    f"[eval] split={name} progress={index + 1}/{len(rows)} "
                    f"accuracy={correct / len(predictions):.4f}",
                    flush=True,
                )

    total = len(predictions)
    correct = sum(row["correct"] for row in predictions)
    parse_failures = sum(row["predicted"] is None for row in predictions)
    latent_activations = sum(row["latent_activated"] for row in predictions)
    latent_steps = sum(int(row["latent_steps"]) for row in predictions)
    oracle_injections = sum(row["oracle_injected"] for row in predictions)
    oracle_latent_steps = sum(int(row["oracle_latent_steps"]) for row in predictions)
    flow_source_injections = sum(row["flow_source_injected"] for row in predictions)
    generated_pixel_images = sum(
        len(row["generated_pixel_images"]) for row in predictions
    )
    if gt_intermediate_in_prompt and (oracle_injections or oracle_latent_steps):
        raise RuntimeError(
            "GT-intermediate prompt evaluation unexpectedly injected oracle latents"
        )
    if gt_intermediate_in_prompt and any(
        int(row["prompt_image_count"]) != 2 for row in predictions
    ):
        raise RuntimeError("GT-intermediate prompt evaluation did not use two images")
    return {
        "split": name,
        "inference_condition": (
            "oracle_latents"
            if use_oracle_latents
            else (
                "gt_intermediate_image_prompt"
                if gt_intermediate_in_prompt
                else "standard_inference"
            )
        ),
        "dataset_path": os.fspath(source_path),
        "predictions_path": os.fspath(output_path),
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "parse_failures": parse_failures,
        "parse_success_rate": (total - parse_failures) / total,
        "latent_activations": latent_activations,
        "latent_activation_rate": latent_activations / total,
        "total_latent_steps": latent_steps,
        "mean_latent_steps": latent_steps / total,
        "oracle_injections": oracle_injections,
        "oracle_injection_rate": oracle_injections / total,
        "total_oracle_latent_steps": oracle_latent_steps,
        "mean_oracle_latent_steps": oracle_latent_steps / total,
        "flow_source_mode": getattr(backend, "flow_source_mode", None),
        "flow_source_injections": flow_source_injections,
        "flow_source_injection_rate": flow_source_injections / total,
        "generated_pixel_images": generated_pixel_images,
        "generated_pixel_images_per_sample": generated_pixel_images / total,
        "gt_intermediate_in_prompt": gt_intermediate_in_prompt,
        "prompt_image_count": 2 if gt_intermediate_in_prompt else 1,
        "oracle_latent_replacement": use_oracle_latents,
        "gold_answers": dict(sorted(answer_counts.items())),
        "predicted_answers": dict(sorted(predicted_counts.items())),
        "by_transform": _summarize_grouped(predictions, "transform_description"),
        "by_shape_C_family": _summarize_grouped(predictions, "shape_C_family"),
    }


def main() -> None:
    args = parse_args()
    if args.oracle_latents and args.gt_intermediate_in_prompt:
        raise ValueError(
            "--oracle-latents and --gt-intermediate-in-prompt are mutually exclusive"
        )
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"--shard-index must be in [0, {args.num_shards}); got {args.shard_index}"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_root = args.image_root.resolve() if args.image_root is not None else None
    parquet_split_paths = _parse_parquet_splits(args.parquet_split)
    if parquet_split_paths and args.image_root is not None:
        raise ValueError("--image-root cannot be combined with --parquet-split")
    backend_kwargs = {
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "max_latent_steps": args.max_latent_steps,
        "dtype": args.dtype,
        "device": args.device,
        "attn_implementation": args.attn_implementation,
        "allow_stage1_latent_inference": (
            args.allow_stage1_latent_inference or args.oracle_latents
        ),
    }
    if hasattr(args, "flow_solver"):
        backend_kwargs["flow_solver"] = args.flow_solver
    if hasattr(args, "flow_seed"):
        backend_kwargs["flow_seed"] = args.flow_seed
    if hasattr(args, "flow_noise_device"):
        backend_kwargs["flow_noise_device"] = args.flow_noise_device
    if hasattr(args, "flow_state_dtype"):
        backend_kwargs["flow_state_dtype"] = args.flow_state_dtype
    if getattr(args, "save_generated_pixels", False):
        backend_kwargs["save_generated_pixels_dir"] = output_dir / "generated_pixels"
    if hasattr(args, "visual_consumer_mode"):
        backend_kwargs["visual_consumer_mode"] = args.visual_consumer_mode
    if hasattr(args, "generated_latent_ablation"):
        backend_kwargs["generated_latent_ablation"] = args.generated_latent_ablation
    if hasattr(args, "generated_latent_ablation_seed"):
        backend_kwargs["generated_latent_ablation_seed"] = (
            args.generated_latent_ablation_seed
        )
    if hasattr(args, "force_oracle_blocks_at_start"):
        if args.force_oracle_blocks_at_start and not args.oracle_latents:
            raise ValueError("--force-oracle-blocks-at-start requires --oracle-latents")
        backend_kwargs["force_oracle_blocks_at_start"] = (
            args.force_oracle_blocks_at_start
        )
    backend = WMVLMBackend(args.model, **backend_kwargs)
    if parquet_split_paths:
        split_paths = parquet_split_paths
        requested_splits = args.splits or list(parquet_split_paths)
    else:
        split_paths = {
            "id_eval": args.eval_json.resolve(),
            "ood": args.held_out_json.resolve(),
        }
        requested_splits = args.splits or ["id_eval", "ood"]
    unknown_splits = set(requested_splits) - set(split_paths)
    if unknown_splits:
        raise ValueError(
            f"Requested splits have no configured source: {sorted(unknown_splits)}"
        )
    metrics: dict[str, Any] = {
        "model": args.model,
        "checkpoint_stage": backend.checkpoint_stage,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "max_latent_steps": args.max_latent_steps,
        "flow_solver": getattr(backend, "flow_solver", None),
        "flow_seed": getattr(backend, "flow_seed", None),
        "flow_noise_device": getattr(backend, "flow_noise_device", None),
        "flow_state_dtype": getattr(backend, "flow_state_dtype_name", None),
        "flow_source_mode": getattr(backend, "flow_source_mode", None),
        "ce_consumer_source": getattr(backend, "ce_consumer_source", None),
        "visual_consumer_mode": getattr(backend, "visual_consumer_mode", None),
        "generated_latent_ablation": getattr(
            backend, "generated_latent_ablation", "none"
        ),
        "generated_latent_ablation_seed": getattr(
            backend, "generated_latent_ablation_seed", None
        ),
        "save_generated_pixels": getattr(args, "save_generated_pixels", False),
        "generated_pixels_dir": (
            str(output_dir / "generated_pixels")
            if getattr(args, "save_generated_pixels", False)
            else None
        ),
        "inference_condition": (
            "oracle_latents"
            if args.oracle_latents
            else (
                "gt_intermediate_image_prompt"
                if args.gt_intermediate_in_prompt
                else "standard_inference"
            )
        ),
        "oracle_latents": args.oracle_latents,
        "gt_intermediate_in_prompt": args.gt_intermediate_in_prompt,
        "oracle_latent_replacement": args.oracle_latents,
        "force_oracle_blocks_at_start": getattr(
            args, "force_oracle_blocks_at_start", False
        ),
        "checkpoint_latent_size": int(backend.model.config.wm_vlm_latent_size),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "image_root": str(image_root) if image_root is not None else None,
        "dataset_format": "parquet" if parquet_split_paths else "json_paths",
        "splits": {},
    }
    for split in requested_splits:
        path = split_paths[split]
        if split in parquet_split_paths:
            rows = _load_parquet_split(
                path,
                args.limit,
                num_shards=args.num_shards,
                shard_index=args.shard_index,
                include_oracle_images=(
                    args.oracle_latents or args.gt_intermediate_in_prompt
                ),
                include_flow_source_image=bool(
                    getattr(backend, "requires_flow_source_image", False)
                ),
            )
        else:
            rows = _load_split(
                path,
                args.limit,
                num_shards=args.num_shards,
                shard_index=args.shard_index,
            )
        metrics["splits"][split] = evaluate_split(
            name=split,
            source_path=path,
            rows=rows,
            backend=backend,
            output_dir=output_dir,
            log_every=args.log_every,
            use_oracle_latents=args.oracle_latents,
            gt_intermediate_in_prompt=args.gt_intermediate_in_prompt,
            image_root=image_root,
            flow_source_image_dir=(
                getattr(args, "flow_source_id_dir", None)
                if split == "id_eval"
                else getattr(args, "flow_source_ood_dir", None)
            ),
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
