"""Convert supported OPT-SAR datasets into one manifest-based layout.

This tool copies/extracts only data selected by --limit. It never modifies the
source dataset. Use --link-mode copy on Windows for the most portable result.
"""

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def stable_split(sample_id, train=90, val=5):
    bucket = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < train else "val" if bucket < train + val else "test"


def transfer(source, destination, mode="copy"):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if mode == "hardlink":
        destination.hardlink_to(source)
    else:
        shutil.copy2(source, destination)


def write_manifest(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} pairs: {path}")


def base_row(sample_id, dataset, opt_path, sar_path, args):
    return {
        "id": sample_id,
        "split": stable_split(sample_id),
        "dataset": dataset,
        "opt_path": opt_path.as_posix(),
        "sar_path": sar_path.as_posix(),
        "opt_sensor": args.opt_sensor,
        "sar_sensor": args.sar_sensor,
        "opt_gsd": args.opt_gsd,
        "sar_gsd": args.sar_gsd,
        "polarization": args.polarization,
        "incidence_angle": None,
        "sar_unit": args.sar_unit,
    }


def prepare_sar1m(args):
    output = Path(args.output)
    rows = []
    with zipfile.ZipFile(args.source) as archive:
        pairs = json.loads(archive.read("paired.json"))
        selected = pairs if args.limit is None else pairs[:args.limit]
        for index, pair in enumerate(selected):
            sample_id = f"sar1m_{Path(pair['sar']).stem}"
            suffix_opt = Path(pair["optical"]).suffix.lower()
            suffix_sar = Path(pair["sar"]).suffix.lower()
            opt_rel = Path("images") / "SAR-1M" / "opt" / f"{sample_id}{suffix_opt}"
            sar_rel = Path("images") / "SAR-1M" / "sar" / f"{sample_id}{suffix_sar}"
            (output / opt_rel).parent.mkdir(parents=True, exist_ok=True)
            (output / sar_rel).parent.mkdir(parents=True, exist_ok=True)
            with archive.open(pair["optical"]) as src, (output / opt_rel).open("wb") as dst:
                shutil.copyfileobj(src, dst)
            with archive.open(pair["sar"]) as src, (output / sar_rel).open("wb") as dst:
                shutil.copyfileobj(src, dst)
            rows.append(base_row(sample_id, "SAR-1M", opt_rel, sar_rel, args))
    write_manifest(rows, output)


def prepare_osdataset(args):
    source = Path(args.source)
    output = Path(args.output)
    opt_files = sorted(source.rglob("opt*.png"))
    rows = []
    for opt in opt_files:
        match = re.fullmatch(r"opt(.+)\.png", opt.name, flags=re.IGNORECASE)
        sar = opt.with_name(f"sar{match.group(1)}.png") if match else None
        if not sar or not sar.exists():
            continue
        region = opt.parent.name.replace(" ", "_")
        sample_id = f"osdataset_{region}_{match.group(1)}"
        opt_rel = Path("images") / "OSDataset" / "opt" / f"{sample_id}.png"
        sar_rel = Path("images") / "OSDataset" / "sar" / f"{sample_id}.png"
        transfer(opt, output / opt_rel, args.link_mode)
        transfer(sar, output / sar_rel, args.link_mode)
        rows.append(base_row(sample_id, "OSDataset", opt_rel, sar_rel, args))
        if args.limit is not None and len(rows) >= args.limit:
            break
    write_manifest(rows, output)


def prepare_mos_ship(args):
    source = Path(args.source)
    output = Path(args.output)
    rows = []
    for opt in sorted(source.rglob("*_rgb.png")):
        sar = opt.with_name(opt.name.replace("_rgb.png", "_sar.png"))
        if not sar.exists():
            continue
        raw_stem = opt.stem[:-4]
        sample_id = f"mosship_{raw_stem}"
        opt_rel = Path("images") / "MOS-Ship" / "opt" / f"{sample_id}.png"
        sar_rel = Path("images") / "MOS-Ship" / "sar" / f"{sample_id}.png"
        transfer(opt, output / opt_rel, args.link_mode)
        transfer(sar, output / sar_rel, args.link_mode)
        row = base_row(sample_id, "MOS-Ship", opt_rel, sar_rel, args)
        row["split"] = "val" if "val" in opt.parts else "train"
        rows.append(row)
        if args.limit is not None and len(rows) >= args.limit:
            break
    write_manifest(rows, output)


def key_for(path, prefix):
    stem = path.stem
    return stem[len(prefix):] if prefix and stem.startswith(prefix) else stem


def prepare_paired_dirs(args):
    source = Path(args.source)
    opt_dir = source / args.opt_dir
    sar_dir = source / args.sar_dir
    output = Path(args.output)
    opt_map = {key_for(p, args.opt_prefix): p for p in opt_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    sar_map = {key_for(p, args.sar_prefix): p for p in sar_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    keys = sorted(set(opt_map).intersection(sar_map))
    if args.limit is not None:
        keys = keys[:args.limit]
    rows = []
    slug = re.sub(r"[^a-z0-9]+", "", args.dataset.lower())
    for index, key in enumerate(keys):
        sample_id = f"{slug}_{index:07d}"
        opt_rel = Path("images") / args.dataset / "opt" / f"{sample_id}{opt_map[key].suffix.lower()}"
        sar_rel = Path("images") / args.dataset / "sar" / f"{sample_id}{sar_map[key].suffix.lower()}"
        transfer(opt_map[key], output / opt_rel, args.link_mode)
        transfer(sar_map[key], output / sar_rel, args.link_mode)
        rows.append(base_row(sample_id, args.dataset, opt_rel, sar_rel, args))
    write_manifest(rows, output)


def add_common(parser):
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--opt-sensor", default="unknown")
    parser.add_argument("--sar-sensor", default="unknown")
    parser.add_argument("--opt-gsd", type=float, default=10.0)
    parser.add_argument("--sar-gsd", type=float, default=10.0)
    parser.add_argument("--polarization", default="unknown")
    parser.add_argument("--sar-unit", default="display_uint8")
    parser.add_argument("--link-mode", choices=("copy", "hardlink"), default="copy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="adapter", required=True)
    for name in ("sar1m-zip", "osdataset", "mos-ship"):
        add_common(sub.add_parser(name))
    generic = sub.add_parser("paired-dirs")
    add_common(generic)
    generic.add_argument("--dataset", required=True)
    generic.add_argument("--opt-dir", required=True)
    generic.add_argument("--sar-dir", required=True)
    generic.add_argument("--opt-prefix", default="")
    generic.add_argument("--sar-prefix", default="")
    args = parser.parse_args()
    {"sar1m-zip": prepare_sar1m, "osdataset": prepare_osdataset,
     "mos-ship": prepare_mos_ship, "paired-dirs": prepare_paired_dirs}[args.adapter](args)


if __name__ == "__main__":
    main()
