#!/usr/bin/env python3
"""Aggregate WM-VLM per-example latent-distance metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


COMPARISONS = (
    "generated_vs_gt",
    "initial_noise_vs_gt",
    "generated_vs_shuffled_gt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--expected-id", type=int, default=None)
    parser.add_argument("--expected-ood", type=int, default=None)
    parser.add_argument("inputs", type=Path, nargs="+")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    examples: dict[tuple[str, int], dict[str, Any]] = {}
    for path in args.inputs:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (str(row["split"]), int(row["index"]))
                if key in examples:
                    raise ValueError(f"Duplicate example {key} from {path}")
                examples[key] = row

    expected = {"id_eval": args.expected_id, "ood": args.expected_ood}
    output: dict[str, Any] = {"splits": {}}
    csv_rows: list[dict[str, Any]] = []
    for split in ("id_eval", "ood"):
        rows = [
            row
            for (row_split, _), row in sorted(examples.items())
            if row_split == split
        ]
        if expected[split] is not None and len(rows) != expected[split]:
            raise ValueError(
                f"{split} count mismatch: expected={expected[split]}, actual={len(rows)}"
            )
        generated = sum(bool(row["latent_generated"]) for row in rows)
        split_payload: dict[str, Any] = {
            "total": len(rows),
            "latent_generated": generated,
            "latent_generation_rate": generated / len(rows),
            "comparisons": {},
        }
        for comparison in COMPARISONS:
            selected = [row[comparison] for row in rows if comparison in row]
            metric_names = sorted(selected[0]) if selected else []
            comparison_payload: dict[str, Any] = {
                "count": len(selected),
                "metrics": {},
            }
            for metric in metric_names:
                values = [float(item[metric]) for item in selected]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Non-finite {split}/{comparison}/{metric}")
                summary = summarize(values)
                comparison_payload["metrics"][metric] = summary
                csv_rows.append(
                    {
                        "split": split,
                        "comparison": comparison,
                        "metric": metric,
                        **summary,
                    }
                )
            split_payload["comparisons"][comparison] = comparison_payload
        output["splits"][split] = split_payload

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)
    print(
        f"[latent-distance-aggregate] json={args.output_json} csv={args.output_csv}",
        flush=True,
    )


if __name__ == "__main__":
    main()
