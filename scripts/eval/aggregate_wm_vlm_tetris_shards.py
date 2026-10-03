#!/usr/bin/env python3
"""Aggregate disjoint WM-VLM Tetris evaluation-shard metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregates: dict[str, dict[str, int | float]] = {}
    seen_shards: set[tuple[str, int]] = set()
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        shard_index = int(payload.get("shard_index", 0))
        for split, metrics in payload["splits"].items():
            shard_key = (split, shard_index)
            if shard_key in seen_shards:
                raise ValueError(f"Duplicate shard {shard_key} in {path}")
            seen_shards.add(shard_key)
            target = aggregates.setdefault(split, {"correct": 0, "total": 0})
            target["correct"] = int(target["correct"]) + int(metrics["correct"])
            target["total"] = int(target["total"]) + int(metrics["total"])
    for metrics in aggregates.values():
        total = int(metrics["total"])
        metrics["accuracy"] = int(metrics["correct"]) / total
    output = {"splits": dict(sorted(aggregates.items()))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
