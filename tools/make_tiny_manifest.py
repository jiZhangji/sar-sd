"""Create a tiny self-contained manifest whose image paths remain valid."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-samples", type=int, default=2)
    parser.add_argument("--val-samples", type=int, default=1)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    wanted = {"train": args.train_samples, "val": args.val_samples}
    selected = {"train": [], "val": []}

    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            split = row.get("split", "train")
            if split not in selected or len(selected[split]) >= wanted[split]:
                continue
            for key in ("opt_path", "sar_path"):
                path = Path(row[key])
                absolute = path if path.is_absolute() else (source.parent / path).resolve()
                if not absolute.is_file():
                    raise FileNotFoundError(f"Missing {key}: {absolute}")
                row[key] = Path(os.path.relpath(absolute, output.parent)).as_posix()
            selected[split].append(row)
            if all(len(selected[name]) >= wanted[name] for name in selected):
                break

    missing = {name: wanted[name] - len(selected[name]) for name in selected if len(selected[name]) < wanted[name]}
    if missing:
        raise RuntimeError(f"Source manifest does not contain enough samples: {missing}")
    rows = selected["train"] + selected["val"]
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
