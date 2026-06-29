"""Automatically prepare the server datasets for Stage-1 OPT-to-SAR training.

Recognized input layout:
  DATASET_ROOT/SAR-1M_full/SAR-1M_DATA.zip
  DATASET_ROOT/whu_opt_sar/{optical.zip,sar.zip}
  DATASET_ROOT/SAR2Opt_full/{A,B}
  DATASET_ROOT/M4-SAR/                 # reported and intentionally skipped

M4-SAR is reserved for Stage 2 domain adaptation and is never added to the
Stage-1 manifest by this script.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def split_for(group_id):
    """Deterministic group-level 90/5/5 split."""
    bucket = int(hashlib.sha1(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 90 else "val" if bucket < 95 else "test"


def relative(path, manifest_dir):
    return Path(os.path.relpath(Path(path).resolve(), manifest_dir.resolve())).as_posix()


def safe_extract(archive_path, output_dir):
    """Extract once and reject ZIP path traversal."""
    marker = output_dir / ".extract_complete"
    if marker.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
            archive.extract(member, output_dir)
    marker.write_text(str(archive_path), encoding="utf-8")


def add_sar1m(dataset_root, prepared_root, rows, limit):
    archive_path = dataset_root / "SAR-1M_full" / "SAR-1M_DATA.zip"
    if not archive_path.is_file():
        print(f"[skip] SAR-1M archive not found: {archive_path}")
        return
    output = prepared_root / "images" / "SAR-1M"
    with zipfile.ZipFile(archive_path) as archive:
        pairs = json.loads(archive.read("paired.json"))
        if limit:
            pairs = pairs[:limit]
        for index, pair in enumerate(pairs, 1):
            sar_member, opt_member = pair["sar"], pair["optical"]
            stem = Path(sar_member).stem
            sample_id = f"sar1m_{stem}"
            group = re.sub(r"_[1-4]$", "", stem)
            opt_path = output / "opt" / f"{sample_id}{Path(opt_member).suffix.lower()}"
            sar_path = output / "sar" / f"{sample_id}{Path(sar_member).suffix.lower()}"
            if not opt_path.exists():
                opt_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(opt_member) as src, opt_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            if not sar_path.exists():
                sar_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(sar_member) as src, sar_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            rows.append(make_row(
                sample_id, split_for(f"sar1m:{group}"), "SAR-1M", opt_path, sar_path,
                prepared_root, "unknown", "unknown", None, None, "unknown",
            ))
            if index % 10000 == 0:
                print(f"[SAR-1M] prepared {index}/{len(pairs)}")
    print(f"[SAR-1M] added {len(pairs)} pairs")


def find_named_root(root, expected):
    candidates = [path for path in root.rglob("*") if path.is_dir() and path.name.lower() == expected.lower()]
    return candidates[0] if candidates else None


def image_map(root):
    return {path.stem: path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}


def add_whu(dataset_root, prepared_root, rows, limit):
    source = dataset_root / "whu_opt_sar"
    optical_zip, sar_zip = source / "optical.zip", source / "sar.zip"
    if not optical_zip.is_file() or not sar_zip.is_file():
        print(f"[skip] WHU archives not found under {source}")
        return
    extracted = prepared_root / "raw" / "WHU-OPT-SAR"
    safe_extract(optical_zip, extracted / "optical_archive")
    safe_extract(sar_zip, extracted / "sar_archive")
    opt_map = image_map(extracted / "optical_archive")
    sar_map = image_map(extracted / "sar_archive")
    keys = sorted(set(opt_map).intersection(sar_map))
    if limit:
        keys = keys[:limit]
    for key in keys:
        sample_id = f"whu_{key}"
        rows.append(make_row(
            sample_id, split_for(f"whu:{key}"), "WHU-OPT-SAR",
            opt_map[key], sar_map[key], prepared_root,
            "GF-1", "GF-3", 5.0, 5.0, "unknown",
        ))
    print(f"[WHU-OPT-SAR] added {len(keys)} pairs")


def add_sar2opt(dataset_root, prepared_root, rows, limit):
    source = dataset_root / "SAR2Opt_full"
    sar_root, opt_root = source / "A", source / "B"
    if not sar_root.is_dir() or not opt_root.is_dir():
        print(f"[skip] SAR2Opt A/B folders not found under {source}")
        return
    sar_map = {path.relative_to(sar_root).as_posix(): path for path in sar_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES}
    opt_map = {path.relative_to(opt_root).as_posix(): path for path in opt_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES}
    keys = sorted(set(sar_map).intersection(opt_map))
    if limit:
        keys = keys[:limit]
    for index, key in enumerate(keys):
        sample_id = f"sar2opt_{index:06d}"
        source_split = Path(key).parts[0].lower() if Path(key).parts else ""
        split = "train" if source_split == "train" else "test" if source_split == "test" else split_for(f"sar2opt:{key}")
        rows.append(make_row(
            sample_id, split, "SAR2Opt", opt_map[key], sar_map[key], prepared_root,
            "Google-Earth", "TerraSAR-X", 1.0, 1.0, "unknown",
        ))
    print(f"[SAR2Opt] added {len(keys)} pairs; ignored duplicate combined train/test folders")


def make_row(sample_id, split, dataset, opt_path, sar_path, manifest_dir,
             opt_sensor, sar_sensor, opt_gsd, sar_gsd, polarization):
    return {
        "id": sample_id,
        "split": split,
        "dataset": dataset,
        "opt_path": relative(opt_path, manifest_dir),
        "sar_path": relative(sar_path, manifest_dir),
        "opt_sensor": opt_sensor,
        "sar_sensor": sar_sensor,
        "opt_gsd": opt_gsd,
        "sar_gsd": sar_gsd,
        "polarization": polarization,
        "incidence_angle": None,
        "sar_unit": "display_uint8",
    }


def validate(rows, manifest_dir):
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise RuntimeError(f"Duplicate id: {row['id']}")
        ids.add(row["id"])
        for key in ("opt_path", "sar_path"):
            if not (manifest_dir / row[key]).resolve().is_file():
                raise FileNotFoundError(f"Missing {key}: {row[key]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit-per-dataset", type=int, default=0,
                        help="0 means all pairs; use a small number for smoke tests")
    parser.add_argument("--skip-sar1m", action="store_true")
    parser.add_argument("--skip-whu", action="store_true")
    parser.add_argument("--skip-sar2opt", action="store_true")
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    limit = args.limit_per_dataset or None
    if not args.skip_sar1m:
        add_sar1m(dataset_root, output_root, rows, limit)
    if not args.skip_whu:
        add_whu(dataset_root, output_root, rows, limit)
    if not args.skip_sar2opt:
        add_sar2opt(dataset_root, output_root, rows, limit)
    if (dataset_root / "M4-SAR").exists():
        print("[M4-SAR] found and intentionally excluded: reserved for Stage 2")
    validate(rows, output_root)
    rows.sort(key=lambda row: (row["dataset"], row["id"]))
    manifest = output_root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {
        "total": len(rows),
        "datasets": Counter(row["dataset"] for row in rows),
        "splits": Counter(row["split"] for row in rows),
    }
    stats = {"total": stats["total"], "datasets": dict(stats["datasets"]), "splits": dict(stats["splits"])}
    (output_root / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
