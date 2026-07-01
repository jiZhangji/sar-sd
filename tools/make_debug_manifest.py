"""Create small debug manifests, optionally filtered by dataset."""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path


def resolve_path(source_dir, value):
    path = Path(value)
    return path if path.is_absolute() else (source_dir / path).resolve()


def rewrite_paths(row, source_dir, output_dir):
    row = dict(row)
    for key in ("opt_path", "sar_path"):
        if not row.get(key):
            continue
        absolute = resolve_path(source_dir, row[key])
        if not absolute.is_file():
            raise FileNotFoundError(f"Missing {key}: {absolute}")
        row[key] = Path(os.path.relpath(absolute, output_dir)).as_posix()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source .jsonl manifest")
    parser.add_argument("--output", required=True, help="Output .jsonl manifest")
    parser.add_argument("--dataset", action="append", help="Keep exact dataset name; can be repeated")
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--val-samples", type=int, default=4)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    wanted = {
        "train": int(args.train_samples),
        "val": int(args.val_samples),
        "test": int(args.test_samples),
    }
    datasets = set(args.dataset or [])
    by_split = defaultdict(list)
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if datasets and row.get("dataset") not in datasets:
                continue
            split = row.get("split", "train")
            if split in wanted and wanted[split] > 0:
                by_split[split].append(row)

    rng = random.Random(args.seed)
    selected = []
    summary = {}
    for split, count in wanted.items():
        rows = by_split.get(split, [])
        if not args.no_shuffle:
            rows = list(rows)
            rng.shuffle(rows)
        if len(rows) < count:
            raise RuntimeError(
                f"Not enough {split} samples after filtering: wanted={count}, found={len(rows)}"
            )
        picked = rows[:count]
        summary[split] = len(picked)
        selected.extend(rewrite_paths(row, source.parent, output.parent) for row in picked)

    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    dataset_text = ",".join(sorted(datasets)) if datasets else "ALL"
    print(json.dumps({
        "output": str(output),
        "dataset_filter": dataset_text,
        "rows": len(selected),
        "splits": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
