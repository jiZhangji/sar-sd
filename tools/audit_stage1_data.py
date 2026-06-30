"""Audit whether Stage-1 prepared data completely matches the raw datasets."""

import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
EXPECTED_DATASETS = {"SAR-1M", "WHU-OPT-SAR", "SAR2Opt"}


def image_stems_in_zip(path):
    with zipfile.ZipFile(path) as archive:
        return {
            Path(name).stem
            for name in archive.namelist()
            if Path(name).suffix.lower() in IMAGE_SUFFIXES
        }


def discover_raw(dataset_root):
    result = {}

    sar1m_zip = dataset_root / "SAR-1M_full" / "SAR-1M_DATA.zip"
    if sar1m_zip.is_file():
        with zipfile.ZipFile(sar1m_zip) as archive:
            pairs = json.loads(archive.read("paired.json"))
            members = set(archive.namelist())
        missing_members = sum(
            pair.get("sar") not in members or pair.get("optical") not in members
            for pair in pairs
        )
        result["SAR-1M"] = {
            "available": len(pairs),
            "raw_pair_keys": {f"sar1m_{Path(pair['sar']).stem}" for pair in pairs},
            "raw_missing_archive_members": missing_members,
            "source": str(sar1m_zip),
        }

    whu_root = dataset_root / "whu_opt_sar"
    optical_zip, sar_zip = whu_root / "optical.zip", whu_root / "sar.zip"
    if optical_zip.is_file() and sar_zip.is_file():
        opt_stems = image_stems_in_zip(optical_zip)
        sar_stems = image_stems_in_zip(sar_zip)
        common = opt_stems & sar_stems
        result["WHU-OPT-SAR"] = {
            "available": len(common),
            "raw_pair_keys": {f"whu_{stem}" for stem in common},
            "unmatched_optical": len(opt_stems - sar_stems),
            "unmatched_sar": len(sar_stems - opt_stems),
            "source": str(whu_root),
        }

    sar2opt_root = dataset_root / "SAR2Opt_full"
    sar_root, opt_root = sar2opt_root / "A", sar2opt_root / "B"
    if sar_root.is_dir() and opt_root.is_dir():
        sar_rel = {
            p.relative_to(sar_root).as_posix()
            for p in sar_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        }
        opt_rel = {
            p.relative_to(opt_root).as_posix()
            for p in opt_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        }
        common = sorted(sar_rel & opt_rel)
        # prepare_stage1_data.py numbers sorted relative paths from zero.
        result["SAR2Opt"] = {
            "available": len(common),
            "raw_pair_keys": {f"sar2opt_{i:06d}" for i in range(len(common))},
            "unmatched_optical": len(opt_rel - sar_rel),
            "unmatched_sar": len(sar_rel - opt_rel),
            "source": str(sar2opt_root),
        }
    return result


def read_manifest(path):
    rows = []
    errors = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {number}: invalid JSON: {exc}")
                continue
            missing = [key for key in ("id", "dataset", "split", "opt_path", "sar_path") if not row.get(key)]
            if missing:
                errors.append(f"line {number}: missing fields {missing}")
            rows.append(row)
    return rows, errors


def audit(dataset_root, prepared_root):
    manifest = prepared_root / "manifest.jsonl"
    if not manifest.is_file():
        return {"status": "FAIL", "errors": [f"manifest not found: {manifest}"]}

    rows, errors = read_manifest(manifest)
    raw = discover_raw(dataset_root)
    by_dataset = defaultdict(list)
    id_counts = Counter()
    pair_counts = Counter()
    missing_files = Counter()
    invalid_splits = Counter()
    for row in rows:
        dataset = str(row.get("dataset", "unknown"))
        by_dataset[dataset].append(row)
        id_counts[str(row.get("id"))] += 1
        pair_counts[(str(row.get("opt_path")), str(row.get("sar_path")))] += 1
        if row.get("split") not in {"train", "val", "test"}:
            invalid_splits[dataset] += 1
        for field in ("opt_path", "sar_path"):
            value = row.get(field)
            path = Path(value) if value else Path("__missing__")
            path = path if path.is_absolute() else prepared_root / path
            if not path.resolve().is_file():
                missing_files[f"{dataset}:{field}"] += 1

    duplicate_ids = sorted(key for key, count in id_counts.items() if count > 1)
    duplicate_pairs = sum(count - 1 for count in pair_counts.values() if count > 1)
    datasets = {}
    complete = True
    for name in sorted(EXPECTED_DATASETS | set(raw) | set(by_dataset)):
        manifest_rows = by_dataset.get(name, [])
        manifest_ids = {str(row.get("id")) for row in manifest_rows}
        info = raw.get(name)
        if info is None:
            datasets[name] = {"raw_available": None, "manifest_count": len(manifest_rows), "complete": False, "note": "raw source not found"}
            complete = False
            continue
        missing_ids = info["raw_pair_keys"] - manifest_ids
        extra_ids = manifest_ids - info["raw_pair_keys"]
        item = {
            "raw_available": info["available"],
            "manifest_count": len(manifest_rows),
            "unique_manifest_ids": len(manifest_ids),
            "missing_from_manifest": len(missing_ids),
            "extra_in_manifest": len(extra_ids),
            "complete": not missing_ids and not extra_ids and len(manifest_rows) == info["available"],
            "source": info["source"],
        }
        for key in ("raw_missing_archive_members", "unmatched_optical", "unmatched_sar"):
            if key in info:
                item[key] = info[key]
        if missing_ids:
            item["missing_examples"] = sorted(missing_ids)[:10]
        if extra_ids:
            item["extra_examples"] = sorted(extra_ids)[:10]
        datasets[name] = item
        complete &= item["complete"]

    m4_count = len(by_dataset.get("M4-SAR", []))
    if m4_count:
        errors.append(f"M4-SAR has {m4_count} Stage-1 rows; it must be reserved for Stage 2")
    complete &= not errors and not duplicate_ids and not duplicate_pairs and not missing_files and not invalid_splits
    return {
        "status": "PASS" if complete else "FAIL",
        "manifest": str(manifest),
        "manifest_total": len(rows),
        "datasets": datasets,
        "splits": dict(Counter(str(row.get("split")) for row in rows)),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_id_examples": duplicate_ids[:10],
        "duplicate_pair_rows": duplicate_pairs,
        "missing_files": dict(missing_files),
        "invalid_splits": dict(invalid_splits),
        "m4_rows_in_stage1": m4_count,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Raw dataset directory")
    parser.add_argument("--prepared-root", required=True, help="stage1_prepared directory")
    parser.add_argument("--report", help="Output report; default: PREPARED_ROOT/audit_report.json")
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    prepared_root = Path(args.prepared_root).resolve()
    report_path = Path(args.report).resolve() if args.report else prepared_root / "audit_report.json"
    report = audit(dataset_root, prepared_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={report_path}")
    sys.exit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
