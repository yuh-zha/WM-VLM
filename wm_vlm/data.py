"""Lazy interleaved manifest dataset and Qwen2.5-VL training collators."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from PIL import Image

from .constants import (
    DEFAULT_REASONING_TEXT,
    WM_VLM_SYSTEM_PROMPT,
)
from .tokens import latent_token_strings


_QUESTION_RE = re.compile(
    r"Question:\s*(.*?)\s*Reasoning so far:\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)
_ANSWER_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)

IMAGINATION_PLACEHOLDER = "[IMAGINATION]"
QWEN_IMAGINATION_INSTRUCTION = (
    "Think step by step. When spatial imagination is needed, output "
    f"{IMAGINATION_PLACEHOLDER} at that point in the reasoning. "
    "Put the final option letter inside <answer> and </answer>."
)


def preprocess_with_qwen_vl_utils(image: Image.Image) -> Image.Image:
    """Apply the pre-processor resize used by the source Tetris implementation."""
    try:
        from qwen_vl_utils import fetch_image
    except ImportError as error:
        raise ImportError(
            "--qwen-vl-utils-preprocess requires qwen-vl-utils in the training environment"
        ) from error
    return fetch_image({"image": image})


@dataclass(frozen=True)
class WMVLMExample:
    id: str
    input_image_paths: tuple[Path, ...]
    helper_image_path: Path
    question: str
    reasoning_text: str
    answer: str
    post_reasoning_text: str = ""
    system_prompt: str = WM_VLM_SYSTEM_PROMPT
    flow_source_image_path: Path | None = None
    helper_image_paths: tuple[Path, ...] = ()
    reasoning_text_blocks: tuple[str, ...] = ()

    @property
    def ordered_helper_image_paths(self) -> tuple[Path, ...]:
        """Return every supervised image in its canonical trace order."""
        return self.helper_image_paths or (self.helper_image_path,)

    @property
    def ordered_reasoning_text_blocks(self) -> tuple[str, ...]:
        """Return the text immediately preceding each supervised image."""
        return self.reasoning_text_blocks or (self.reasoning_text,)


class WMVLMManifestDataset(torch.utils.data.Dataset):
    """Training split backed by lazy JSONL byte offsets."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        root_dir: str | Path | None = None,
        expected_examples: int | None = None,
        minimum_input_images: int = 1,
        exact_input_images: int | None = None,
        expected_source_format: str | None = None,
        task_name: str = "paper dataset",
        flow_source_image_dir: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir) if root_dir else self.manifest_path.parent
        self.minimum_input_images = int(minimum_input_images)
        self.exact_input_images = (
            None if exact_input_images is None else int(exact_input_images)
        )
        self.expected_source_format = expected_source_format
        self.task_name = str(task_name).upper()
        self.flow_source_image_dir = (
            None
            if flow_source_image_dir is None
            else Path(flow_source_image_dir).resolve()
        )
        if self.minimum_input_images < 1:
            raise ValueError("minimum_input_images must be positive")
        if self.exact_input_images is not None and self.exact_input_images < 1:
            raise ValueError("exact_input_images must be positive when set")
        if (
            self.exact_input_images is not None
            and self.exact_input_images < self.minimum_input_images
        ):
            raise ValueError("exact_input_images cannot be smaller than minimum_input_images")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"{self.task_name} manifest not found: {self.manifest_path}")
        if (
            self.flow_source_image_dir is not None
            and not self.flow_source_image_dir.is_dir()
        ):
            raise FileNotFoundError(
                f"{self.task_name} flow-source image directory not found: "
                f"{self.flow_source_image_dir}"
            )

        self._offsets: list[int] = []
        self.total_helper_images = 0
        self.max_helper_images_per_example = 0
        self.total_rows = 0
        with self.manifest_path.open("rb") as manifest:
            while True:
                offset = manifest.tell()
                line = manifest.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                self.total_rows += 1
                row = json.loads(line)
                row_id = row.get("id", "<unknown>")
                if (
                    self.expected_source_format is not None
                    and row.get("source_format") != self.expected_source_format
                ):
                    raise ValueError(
                        f"{self.task_name} row {row_id!r} has source_format="
                        f"{row.get('source_format')!r}; expected {self.expected_source_format!r}"
                    )
                input_image_count = len(_source_image_segments(row, task_name=self.task_name))
                if (
                    self.exact_input_images is not None
                    and input_image_count != self.exact_input_images
                ):
                    raise ValueError(
                        f"{self.task_name} row {row_id!r} has {input_image_count} input images; "
                        f"expected exactly {self.exact_input_images}"
                    )
                if input_image_count >= self.minimum_input_images:
                    self._offsets.append(offset)
                    segments = row["segments"]
                    assistant_start = int(row["assistant_start"])
                    helper_count = sum(
                        segment.get("type") == "image"
                        and bool(segment.get("loss", False))
                        for segment in segments[assistant_start:]
                    )
                    if helper_count < 1:
                        raise ValueError(
                            f"{self.task_name} row {row_id!r} has no supervised helper image"
                        )
                    self.total_helper_images += helper_count
                    self.max_helper_images_per_example = max(
                        self.max_helper_images_per_example,
                        helper_count,
                    )

        if expected_examples is not None and len(self._offsets) != expected_examples:
            raise ValueError(
                f"{self.task_name} training count mismatch: "
                f"expected {expected_examples}, found {len(self._offsets)} in {self.manifest_path}. "
                "Use the complete task-specific schema-v4 manifest."
            )
        if not self._offsets:
            raise ValueError(
                f"No eligible {self.task_name} examples found in {self.manifest_path}"
            )

    @property
    def skipped_short_rows(self) -> int:
        return self.total_rows - len(self._offsets)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> WMVLMExample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        with self.manifest_path.open("rb") as manifest:
            manifest.seek(self._offsets[index])
            row = json.loads(manifest.readline())
        example = parse_manifest_row(
            row,
            root_dir=self.root_dir,
            minimum_input_images=self.minimum_input_images,
            exact_input_images=self.exact_input_images,
            task_name=self.task_name,
        )
        if self.flow_source_image_dir is None:
            return example
        if len(example.input_image_paths) != 1:
            raise ValueError(
                "Image-embedding flow sources require exactly one question image; "
                f"example {example.id!r} has {len(example.input_image_paths)}"
            )
        flow_source_path = (
            self.flow_source_image_dir / example.input_image_paths[0].name
        )
        if not flow_source_path.is_file():
            raise FileNotFoundError(
                f"Flow-source query crop not found for example {example.id!r}: "
                f"{flow_source_path}"
            )
        return replace(example, flow_source_image_path=flow_source_path)


def _source_image_segments(
    row: dict[str, Any],
    *,
    task_name: str = "paper dataset",
) -> list[dict[str, Any]]:
    segments = row.get("segments")
    assistant_start = row.get("assistant_start")
    schema_version = row.get("schema_version")
    if schema_version not in {4, 5} or not isinstance(segments, list):
        raise ValueError(
            f"{task_name} row {row.get('id', '<unknown>')!r} is not schema-v4/v5"
        )
    if not isinstance(assistant_start, int) or not 0 <= assistant_start <= len(segments):
        raise ValueError(
            f"{task_name} row {row.get('id', '<unknown>')!r} has invalid assistant_start"
        )
    return [
        segment
        for segment in segments[:assistant_start]
        if segment.get("type") == "image" and not segment.get("loss", False)
    ]


def parse_manifest_row(
    row: dict[str, Any],
    *,
    root_dir: str | Path,
    minimum_input_images: int = 1,
    exact_input_images: int | None = None,
    task_name: str = "paper dataset",
) -> WMVLMExample:
    source_images = _source_image_segments(row, task_name=task_name)
    if len(source_images) < minimum_input_images:
        raise ValueError(
            f"{task_name} row {row.get('id', '<unknown>')!r} has only "
            f"{len(source_images)} input images"
        )
    if exact_input_images is not None and len(source_images) != exact_input_images:
        raise ValueError(
            f"{task_name} row {row.get('id', '<unknown>')!r} has "
            f"{len(source_images)} input images; expected exactly {exact_input_images}"
        )
    segments = row["segments"]
    assistant_start = int(row["assistant_start"])
    helper_images = [
        segment
        for segment in segments[assistant_start:]
        if segment.get("type") == "image" and segment.get("loss", False)
    ]
    schema_version = int(row["schema_version"])
    if not helper_images or (schema_version == 4 and len(helper_images) != 1):
        raise ValueError(
            f"{task_name} row {row.get('id', '<unknown>')!r} must have "
            f"{'one' if schema_version == 4 else 'at least one'} supervised helper image; "
            f"found {len(helper_images)}"
        )

    source_text = "".join(
        str(segment.get("text", ""))
        for segment in segments[:assistant_start]
        if segment.get("type") == "text"
    ).strip()
    question_match = _QUESTION_RE.search(source_text)
    if question_match is None:
        raise ValueError(
            f"Could not parse question from {task_name} row "
            f"{row.get('id', '<unknown>')!r}"
        )
    question = question_match.group(1).strip()

    target_segments = segments[assistant_start:]
    helper_offsets = [
        index
        for index, segment in enumerate(target_segments)
        if segment.get("type") == "image" and segment.get("loss", False)
    ]
    reasoning_text_blocks: list[str] = []
    cursor = 0
    for helper_offset in helper_offsets:
        reasoning_text_blocks.append(
            "".join(
                str(segment.get("text", ""))
                for segment in target_segments[cursor:helper_offset]
                if segment.get("type") == "text" and segment.get("loss", False)
            )
        )
        cursor = helper_offset + 1
    post_helper_texts = [
        str(segment.get("text", ""))
        for segment in target_segments[cursor:]
        if segment.get("type") == "text" and segment.get("loss", False)
    ]
    target_texts = [
        str(segment.get("text", ""))
        for segment in target_segments
        if segment.get("type") == "text" and segment.get("loss", False)
    ]
    answer_match = _ANSWER_RE.search("".join(target_texts))
    answer = "" if answer_match is None else answer_match.group(1).strip()
    if not answer:
        raise ValueError(
            f"Could not parse answer from {task_name} row "
            f"{row.get('id', '<unknown>')!r}"
        )
    # Preserve established normalization for multiple-choice manifests.
    if re.fullmatch(r"[A-E]", answer, flags=re.IGNORECASE):
        answer = answer.upper()
    reasoning = reasoning_text_blocks[0] or DEFAULT_REASONING_TEXT
    post_reasoning = _ANSWER_RE.sub("", "".join(post_helper_texts)).strip()

    def resolve(path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else Path(root_dir) / candidate

    system_prompt = WM_VLM_SYSTEM_PROMPT
    if row.get("use_manifest_system_prompt") is True:
        manifest_system_prompt = str(row.get("system_prompt", "")).strip()
        if not manifest_system_prompt:
            raise ValueError(
                f"{task_name} row {row.get('id', '<unknown>')!r} enables "
                "use_manifest_system_prompt but has no system_prompt"
            )
        system_prompt = manifest_system_prompt

    return WMVLMExample(
        id=str(row.get("id", "")),
        input_image_paths=tuple(
            resolve(str(segment["path"])) for segment in source_images
        ),
        helper_image_path=resolve(str(helper_images[0]["path"])),
        question=question,
        reasoning_text=reasoning,
        answer=answer,
        post_reasoning_text=post_reasoning,
        system_prompt=system_prompt,
        helper_image_paths=tuple(
            resolve(str(segment["path"])) for segment in helper_images
        ),
        reasoning_text_blocks=tuple(reasoning_text_blocks),
    )


def _validate_prompt_style(prompt_style: str) -> str:
    if prompt_style not in {"wm_vlm", "tetris", "tetris_exact"}:
        raise ValueError(
            f"Unsupported prompt_style={prompt_style!r}; expected 'wm_vlm', "
            "'tetris', or 'tetris_exact'"
        )
    return prompt_style


def build_user_messages(
    question: str,
    num_images: int,
    *,
    prompt_style: str = "wm_vlm",
    system_prompt: str = WM_VLM_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """Build either the established WM-VLM prompt or task-native Tetris ordering."""
    prompt_style = _validate_prompt_style(prompt_style)
    images: list[dict[str, str]] = [{"type": "image"} for _ in range(num_images)]
    if prompt_style in {"tetris", "tetris_exact"}:
        # The task-native format has no system turn and puts text before images.
        content = [{"type": "text", "text": question.strip()}, *images]
        return [{"role": "user", "content": content}]
    content = [*images, {"type": "text", "text": question.strip()}]
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": content},
    ]


def build_training_messages(
    example: WMVLMExample,
    latent_size: int,
    *,
    prompt_style: str = "wm_vlm",
    token_style: str = "wm_vlm",
) -> list[dict[str, Any]]:
    prompt_style = _validate_prompt_style(prompt_style)
    latent_token, latent_start, latent_end = latent_token_strings(token_style)
    helper_paths = example.ordered_helper_image_paths
    reasoning_blocks = example.ordered_reasoning_text_blocks
    if len(reasoning_blocks) != len(helper_paths):
        raise ValueError(
            f"Example {example.id!r} has {len(reasoning_blocks)} reasoning blocks "
            f"for {len(helper_paths)} helper images"
        )
    latent_block = latent_start + latent_token * latent_size + latent_end
    if prompt_style in {"tetris", "tetris_exact"}:
        if len(helper_paths) != 1:
            raise ValueError("Multi-block training currently requires prompt_style='wm_vlm'")
        # Mirror SFTTetrisDataset: no inserted newlines around the latent block,
        # lower-case answer with a leading space, and text-first user content.
        post_reasoning = example.post_reasoning_text.strip()
        if prompt_style == "tetris_exact":
            # Native SFTTetrisDataset concatenates post_visual_text_think and
            # text_think directly, with no separator inserted between them.
            post_reasoning = post_reasoning.replace("\n", "")
        assistant = (
            example.reasoning_text.rstrip()
            + latent_block
            + post_reasoning
            + f"<answer> {example.answer.lower()}</answer>"
        )
    else:
        trace_parts: list[str] = []
        for reasoning_text in reasoning_blocks:
            if reasoning_text.strip():
                trace_parts.append(reasoning_text.rstrip())
            trace_parts.append(latent_block)
        assistant = "\n".join(trace_parts)
        if example.post_reasoning_text:
            assistant += "\n" + example.post_reasoning_text.strip()
        assistant += f"\n<answer>{example.answer}</answer>"
    return build_user_messages(
        example.question,
        len(example.input_image_paths),
        prompt_style=prompt_style,
        system_prompt=example.system_prompt,
    ) + [{"role": "assistant", "content": assistant}]


def build_direct_sft_messages(example: WMVLMExample) -> list[dict[str, Any]]:
    """Build the answer-only stock-Qwen control with no WM-VLM tokens or helper image."""
    content: list[dict[str, str]] = [
        {"type": "image"} for _ in example.input_image_paths
    ]
    content.append(
        {
            "type": "text",
            "text": example.question.strip() + "\nAnswer with the option letter only.",
        }
    )
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": example.answer},
    ]


def build_imagination_user_prompt(question: str) -> str:
    """Build the text-only-imagination prompt shared by training and evaluation."""
    return question.strip() + "\n" + QWEN_IMAGINATION_INSTRUCTION


def build_imagination_sft_messages(
    example: WMVLMExample,
) -> list[dict[str, Any]]:
    """Replace the supervised helper image in a full trace with a text marker.

    The source/problem images remain image inputs.  The helper image is never opened;
    its manifest position is represented by the literal ``[IMAGINATION]`` token string.
    """
    content: list[dict[str, str]] = [
        {"type": "image"} for _ in example.input_image_paths
    ]
    content.append(
        {"type": "text", "text": build_imagination_user_prompt(example.question)}
    )
    assistant = example.reasoning_text.rstrip()
    assistant += f"\n{IMAGINATION_PLACEHOLDER}"
    if example.post_reasoning_text:
        assistant += "\n" + example.post_reasoning_text.strip()
    assistant += f"\n<answer>{example.answer}</answer>"
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": assistant},
    ]


def build_interleaved_image_placeholder_sft_messages(
    example: WMVLMExample,
) -> list[dict[str, Any]]:
    """Build ordinary SFT messages by replacing each target image in place.

    Unlike ``build_imagination_sft_messages``, this matched control neither adds
    an imagination instruction to the question nor drops the manifest-provided
    system prompt.  It preserves the source reasoning trace and substitutes the
    literal text ``[IMAGINATION]`` at every supervised-image position.
    """
    helper_paths = example.ordered_helper_image_paths
    reasoning_blocks = example.ordered_reasoning_text_blocks
    if len(reasoning_blocks) != len(helper_paths):
        raise ValueError(
            f"Example {example.id!r} has {len(reasoning_blocks)} reasoning blocks "
            f"for {len(helper_paths)} helper images"
        )
    user_content: list[dict[str, str]] = [
        {"type": "image"} for _ in example.input_image_paths
    ]
    user_content.append({"type": "text", "text": example.question.strip()})

    assistant_parts: list[str] = []
    for reasoning_text in reasoning_blocks:
        if reasoning_text:
            assistant_parts.append(reasoning_text)
        # Keep the replacement as its own textual block. Without a trailing
        # newline, BPE can merge the closing bracket with the following
        # ``<think>`` boundary and the marker is no longer independently encoded.
        assistant_parts.append(IMAGINATION_PLACEHOLDER + "\n")
    assistant = "".join(assistant_parts)
    if example.post_reasoning_text:
        assistant += example.post_reasoning_text.strip()
    assistant += f"\n<answer>{example.answer}</answer>"
    return [
        {"role": "system", "content": example.system_prompt.strip()},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant},
    ]


def build_imagination_with_helper_image_sft_messages(
    example: WMVLMExample,
) -> list[dict[str, Any]]:
    """Insert the GT helper image after ``[IMAGINATION]`` during Qwen SFT.

    The assistant-side image is context rather than a generation target: its
    vision placeholder tokens are masked by the matching collator, while all
    following reasoning and answer tokens can causally attend to its embeddings.
    """
    helper_paths = example.ordered_helper_image_paths
    if len(helper_paths) != 1:
        raise ValueError(
            "The Tetris Qwen helper-image control requires exactly one supervised "
            f"helper image; example {example.id!r} has {len(helper_paths)}"
        )
    content: list[dict[str, str]] = [
        {"type": "image"} for _ in example.input_image_paths
    ]
    content.append(
        {"type": "text", "text": build_imagination_user_prompt(example.question)}
    )
    pre_image_text = example.reasoning_text.rstrip() + f"\n{IMAGINATION_PLACEHOLDER}"
    post_image_text = ""
    if example.post_reasoning_text:
        post_image_text += "\n" + example.post_reasoning_text.strip()
    post_image_text += f"\n<answer>{example.answer}</answer>"
    return [
        {"role": "user", "content": content},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": pre_image_text},
                {"type": "image"},
                {"type": "text", "text": post_image_text},
            ],
        },
    ]


def find_last_subsequence(sequence: list[int], pattern: list[int]) -> int:
    if not pattern:
        raise ValueError("pattern must not be empty")
    for start in range(len(sequence) - len(pattern), -1, -1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    raise ValueError(f"Token pattern {pattern} was not found")


class WMVLMDataCollator:
    """Build one-sample Qwen batches and, in stage 1, the helper target."""

    def __init__(
        self,
        processor: Any,
        *,
        stage: str,
        latent_size: int,
        helper_image_processor: Any | None = None,
        flow_source_image_processor: Any | None = None,
        prompt_style: str = "wm_vlm",
        token_style: str = "wm_vlm",
        padding: str = "longest",
        max_length: int = 4096,
        qwen_vl_utils_preprocess: bool = False,
        problem_qwen_vl_utils_preprocess: bool | None = None,
        helper_qwen_vl_utils_preprocess: bool | None = None,
        flow_source_qwen_vl_utils_preprocess: bool | None = None,
        pixel_reconstruction: bool = False,
        pixel_patch_size: int | None = None,
        spatial_merge_size: int | None = None,
    ) -> None:
        if stage not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported WM-VLM stage: {stage}")
        self.processor = processor
        self.helper_image_processor = (
            processor.image_processor
            if helper_image_processor is None
            else helper_image_processor
        )
        self.flow_source_image_processor = (
            self.helper_image_processor
            if flow_source_image_processor is None
            else flow_source_image_processor
        )
        self.stage = stage
        self.latent_size = int(latent_size)
        self.prompt_style = _validate_prompt_style(prompt_style)
        self.token_style = token_style
        latent_token, _, _ = latent_token_strings(token_style)
        if padding not in {"longest", "max_length"}:
            raise ValueError(
                f"Unsupported padding={padding!r}; expected 'longest' or 'max_length'"
            )
        if max_length < 1:
            raise ValueError(f"max_length must be positive, got {max_length}")
        self.padding = padding
        self.max_length = int(max_length)
        self.qwen_vl_utils_preprocess = bool(qwen_vl_utils_preprocess)
        self.problem_qwen_vl_utils_preprocess = (
            self.qwen_vl_utils_preprocess
            if problem_qwen_vl_utils_preprocess is None
            else bool(problem_qwen_vl_utils_preprocess)
        )
        self.helper_qwen_vl_utils_preprocess = (
            self.qwen_vl_utils_preprocess
            if helper_qwen_vl_utils_preprocess is None
            else bool(helper_qwen_vl_utils_preprocess)
        )
        self.flow_source_qwen_vl_utils_preprocess = (
            self.qwen_vl_utils_preprocess
            if flow_source_qwen_vl_utils_preprocess is None
            else bool(flow_source_qwen_vl_utils_preprocess)
        )
        self.pixel_reconstruction = bool(pixel_reconstruction)
        self.pixel_patch_size = (
            None if pixel_patch_size is None else int(pixel_patch_size)
        )
        self.spatial_merge_size = (
            None if spatial_merge_size is None else int(spatial_merge_size)
        )
        if self.pixel_reconstruction and (
            self.pixel_patch_size is None
            or self.pixel_patch_size < 1
            or self.spatial_merge_size is None
            or self.spatial_merge_size < 1
        ):
            raise ValueError(
                "Pixel reconstruction requires positive pixel_patch_size and "
                "spatial_merge_size"
            )
        self.latent_token_id = int(processor.tokenizer.convert_tokens_to_ids(latent_token))
        self.assistant_marker_ids = processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self.assistant_end_ids = processor.tokenizer.encode(
            "<|im_end|>", add_special_tokens=False
        )

    @staticmethod
    def _open_rgb(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    @staticmethod
    def _resize_rgb_target(
        image: Image.Image,
        *,
        height: int,
        width: int,
        resample: int,
    ) -> torch.Tensor:
        resized = image.resize((width, height), resample=resample)
        pixels = np.array(resized, dtype=np.float32, copy=True)
        if pixels.shape != (height, width, 3):
            raise ValueError(
                "RGB reconstruction target has unexpected shape "
                f"{pixels.shape}; expected {(height, width, 3)}"
            )
        return torch.from_numpy(pixels).permute(2, 0, 1).div_(255.0)

    def __call__(self, examples: list[WMVLMExample]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("WM-VLM collator received an empty batch")
        block_counts = [len(example.ordered_helper_image_paths) for example in examples]
        if (self.stage == "stage2" or any(count > 1 for count in block_counts)) and len(examples) != 1:
            raise ValueError(
                "WM-VLM multi-block/continuous-latent training requires per-device batch size 1; "
                f"received {len(examples)} examples"
            )
        images = [
            self._open_rgb(path)
            for example in examples
            for path in example.input_image_paths
        ]
        if self.problem_qwen_vl_utils_preprocess:
            images = [preprocess_with_qwen_vl_utils(image) for image in images]
        rendered = [
            self.processor.apply_chat_template(
                build_training_messages(
                    example,
                    self.latent_size,
                    prompt_style=self.prompt_style,
                    token_style=self.token_style,
                ),
                tokenize=False,
                add_generation_prompt=False,
            )
            for example in examples
        ]
        batch = self.processor(
            text=rendered,
            images=images,
            padding=self.padding,
            max_length=self.max_length if self.padding == "max_length" else None,
            truncation=self.padding == "max_length",
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        for row_index, row in enumerate(input_ids.tolist()):
            marker_start = find_last_subsequence(row, self.assistant_marker_ids)
            assistant_start = marker_start + len(self.assistant_marker_ids)
            labels[row_index, :assistant_start] = -100
            if self.prompt_style in {"tetris", "tetris_exact"}:
                assistant_end = find_last_subsequence(
                    row,
                    self.assistant_end_ids,
                )
                labels[row_index, assistant_end:] = -100
        labels[batch["attention_mask"] == 0] = -100
        labels[input_ids == self.latent_token_id] = -100
        batch["labels"] = labels

        latent_counts = (input_ids == self.latent_token_id).sum(dim=1)
        expected_latent_counts = torch.tensor(
            [count * self.latent_size for count in block_counts],
            device=latent_counts.device,
            dtype=latent_counts.dtype,
        )
        if not torch.equal(latent_counts, expected_latent_counts):
            raise ValueError(
                "Latent-token count does not match the canonical helper blocks: "
                f"expected={expected_latent_counts.tolist()}, got={latent_counts.tolist()}"
            )
        batch["latent_block_counts"] = torch.tensor(block_counts, dtype=torch.long)

        if self.stage == "stage1":
            helpers = [
                self._open_rgb(path)
                for example in examples
                for path in example.ordered_helper_image_paths
            ]
            if self.helper_qwen_vl_utils_preprocess:
                helpers = [preprocess_with_qwen_vl_utils(image) for image in helpers]
            helper_batch = self.helper_image_processor(
                images=helpers,
                return_tensors="pt",
            )
            batch["pixel_values_latent"] = helper_batch["pixel_values"]
            batch["image_grid_thw_latent"] = helper_batch["image_grid_thw"]
            if self.pixel_reconstruction:
                image_grid_thw = helper_batch["image_grid_thw"]
                if image_grid_thw.shape != (len(helpers), 3):
                    raise ValueError(
                        "Pixel target grids must have shape [images, 3], got "
                        f"{tuple(image_grid_thw.shape)}"
                    )
                if not bool(torch.all(image_grid_thw[:, 0] == 1)):
                    raise ValueError(
                        "Pixel reconstruction supports static helper images only"
                    )
                assert self.spatial_merge_size is not None
                spatial_grid = image_grid_thw[:, 1:]
                if bool((spatial_grid % self.spatial_merge_size != 0).any()):
                    raise ValueError(
                        "Helper grids are not divisible by spatial_merge_size="
                        f"{self.spatial_merge_size}: {spatial_grid.tolist()}"
                    )
                reconstruction_grid = spatial_grid // self.spatial_merge_size
                if not bool(torch.all(reconstruction_grid == reconstruction_grid[:1])):
                    raise ValueError(
                        "Pixel reconstruction requires a common helper grid per batch: "
                        f"{reconstruction_grid.tolist()}"
                    )
                assert self.pixel_patch_size is not None
                target_height = int(reconstruction_grid[0, 0]) * self.pixel_patch_size
                target_width = int(reconstruction_grid[0, 1]) * self.pixel_patch_size
                resample = int(
                    getattr(
                        self.helper_image_processor,
                        "resample",
                        Image.Resampling.BICUBIC,
                    )
                )
                batch["pixel_reconstruction_targets"] = torch.stack(
                    [
                        self._resize_rgb_target(
                            image,
                            height=target_height,
                            width=target_width,
                            resample=resample,
                        )
                        for image in helpers
                    ],
                    dim=0,
                )
                batch["pixel_reconstruction_grid_hw"] = reconstruction_grid.to(
                    dtype=torch.long
                )
        source_paths = [example.flow_source_image_path for example in examples]
        if any(path is not None for path in source_paths):
            if not all(path is not None for path in source_paths):
                raise ValueError(
                    "A batch cannot mix examples with and without flow-source images"
                )
            sources = [self._open_rgb(path) for path in source_paths if path is not None]
            if self.flow_source_qwen_vl_utils_preprocess:
                sources = [preprocess_with_qwen_vl_utils(image) for image in sources]
            source_batch = self.flow_source_image_processor(
                images=sources,
                return_tensors="pt",
            )
            batch["pixel_values_flow_source"] = source_batch["pixel_values"]
            batch["image_grid_thw_flow_source"] = source_batch["image_grid_thw"]
        return dict(batch)


class QwenDirectSFTDataCollator:
    """Build answer-only stock-Qwen batches without latent or helper-image inputs."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.assistant_marker_ids = processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    @staticmethod
    def _open_rgb(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __call__(self, examples: list[WMVLMExample]) -> dict[str, torch.Tensor]:
        if len(examples) != 1:
            raise ValueError(
                "The matched direct-SFT control uses per-device batch size 1; "
                f"received {len(examples)} examples"
            )
        example = examples[0]
        images = [self._open_rgb(path) for path in example.input_image_paths]
        rendered = self.processor.apply_chat_template(
            build_direct_sft_messages(example),
            tokenize=False,
            add_generation_prompt=False,
        )
        batch = self.processor(
            text=[rendered],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        marker_start = find_last_subsequence(
            input_ids[0].tolist(),
            self.assistant_marker_ids,
        )
        assistant_start = marker_start + len(self.assistant_marker_ids)
        labels = input_ids.clone()
        labels[:, :assistant_start] = -100
        labels[batch["attention_mask"] == 0] = -100
        if not bool(torch.any(labels != -100)):
            raise ValueError("Direct-SFT batch has no supervised assistant tokens")
        batch["labels"] = labels
        return dict(batch)

class QwenImaginationSFTDataCollator(QwenDirectSFTDataCollator):
    """Build full-trace Qwen batches with the helper image replaced at read time."""

    def __call__(self, examples: list[WMVLMExample]) -> dict[str, torch.Tensor]:
        if len(examples) != 1:
            raise ValueError(
                "The imagination-SFT control uses per-device batch size 1; "
                f"received {len(examples)} examples"
            )
        example = examples[0]
        images = [self._open_rgb(path) for path in example.input_image_paths]
        rendered = self.processor.apply_chat_template(
            build_imagination_sft_messages(example),
            tokenize=False,
            add_generation_prompt=False,
        )
        batch = self.processor(
            text=[rendered],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        marker_start = find_last_subsequence(
            input_ids[0].tolist(),
            self.assistant_marker_ids,
        )
        assistant_start = marker_start + len(self.assistant_marker_ids)
        labels = input_ids.clone()
        labels[:, :assistant_start] = -100
        labels[batch["attention_mask"] == 0] = -100
        if not bool(torch.any(labels != -100)):
            raise ValueError("Imagination-SFT batch has no supervised assistant tokens")
        batch["labels"] = labels
        return dict(batch)


class QwenInterleavedImagePlaceholderSFTDataCollator(
    QwenDirectSFTDataCollator
):
    """Build ordinary-SFT batches with target images replaced by text markers."""

    def __call__(self, examples: list[WMVLMExample]) -> dict[str, torch.Tensor]:
        if len(examples) != 1:
            raise ValueError(
                "The interleaved image-placeholder SFT control uses per-device "
                f"batch size 1; received {len(examples)} examples"
            )
        example = examples[0]
        images = [self._open_rgb(path) for path in example.input_image_paths]
        rendered = self.processor.apply_chat_template(
            build_interleaved_image_placeholder_sft_messages(example),
            tokenize=False,
            add_generation_prompt=False,
        )
        batch = self.processor(
            text=[rendered],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        marker_start = find_last_subsequence(
            input_ids[0].tolist(),
            self.assistant_marker_ids,
        )
        assistant_start = marker_start + len(self.assistant_marker_ids)
        labels = input_ids.clone()
        labels[:, :assistant_start] = -100
        labels[batch["attention_mask"] == 0] = -100
        if not bool(torch.any(labels != -100)):
            raise ValueError(
                "Interleaved image-placeholder SFT batch has no supervised "
                "assistant tokens"
            )
        batch["labels"] = labels
        return dict(batch)


class QwenImaginationWithHelperImageSFTDataCollator(
    QwenDirectSFTDataCollator
):
    """Expose the GT helper image after the imagination marker during training."""

    def __init__(self, processor: Any) -> None:
        super().__init__(processor)
        token_strings = ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>")
        self.vision_special_token_ids = tuple(
            int(processor.tokenizer.convert_tokens_to_ids(token))
            for token in token_strings
        )
        if len(set(self.vision_special_token_ids)) != len(token_strings):
            raise ValueError(
                "Qwen tokenizer does not expose distinct vision boundary/pad tokens: "
                f"{dict(zip(token_strings, self.vision_special_token_ids))}"
            )

    def __call__(self, examples: list[WMVLMExample]) -> dict[str, torch.Tensor]:
        if len(examples) != 1:
            raise ValueError(
                "The helper-image imagination-SFT control uses per-device batch size 1; "
                f"received {len(examples)} examples"
            )
        example = examples[0]
        helper_paths = example.ordered_helper_image_paths
        if len(helper_paths) != 1:
            raise ValueError(
                "The Tetris Qwen helper-image control requires exactly one supervised "
                f"helper image; example {example.id!r} has {len(helper_paths)}"
            )
        # Qwen consumes images in the same order as their chat-template placeholders:
        # source/problem image(s) first, then the assistant-side GT helper image.
        images = [self._open_rgb(path) for path in example.input_image_paths]
        images.append(self._open_rgb(helper_paths[0]))
        rendered = self.processor.apply_chat_template(
            build_imagination_with_helper_image_sft_messages(example),
            tokenize=False,
            add_generation_prompt=False,
        )
        batch = self.processor(
            text=[rendered],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        marker_start = find_last_subsequence(
            input_ids[0].tolist(),
            self.assistant_marker_ids,
        )
        assistant_start = marker_start + len(self.assistant_marker_ids)
        labels = input_ids.clone()
        labels[:, :assistant_start] = -100
        labels[batch["attention_mask"] == 0] = -100
        # The helper image is supplied as causal context only.  Mask its complete
        # vision placeholder span so the objective never asks Qwen to emit an image.
        for token_id in self.vision_special_token_ids:
            labels[input_ids == token_id] = -100
        if not bool(torch.any(labels != -100)):
            raise ValueError(
                "Helper-image imagination-SFT batch has no supervised assistant tokens"
            )
        batch["labels"] = labels
        return dict(batch)
