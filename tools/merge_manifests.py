"""Merge dataset manifests and rewrite image paths relative to the output file."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    count = 0
    with output.open("w", encoding="utf-8") as dst:
        for manifest_name in args.inputs:
            manifest = Path(manifest_name).resolve()
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["id"] in seen:
                    raise ValueError(f"Duplicate sample id: {row['id']}")
                seen.add(row["id"])
                for key in ("opt_path", "sar_path"):
                    absolute = (manifest.parent / row[key]).resolve()
                    row[key] = absolute.relative_to(output.parent).as_posix()
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    print(f"merged {count} samples into {output}")


if __name__ == "__main__":
    main()
