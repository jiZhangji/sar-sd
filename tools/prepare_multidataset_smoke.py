"""Download/import a tiny, traceable multi-dataset OPT-SAR smoke set.

Automatic sources:
  * SEN1-2: official TUM FTP, two pairs.
  * SAR2Opt: public Hugging Face mirror, two pairs.

3MOS uses Baidu NetDisk and SAR-1M is a gated 76.6 GB ZIP. Their raw samples
can be imported with --three-mos-root / --sar-one-m-root after manual access.
No preview figure is presented as a raw training sample.
"""

import argparse
import json
import shutil
from ftplib import FTP
from pathlib import Path

from huggingface_hub import hf_hub_download


def copy_image(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def download_sen12(output, count):
    rows = []
    ftp = FTP("dataserv.ub.tum.de", timeout=90)
    ftp.login("m1436631", "m1436631")
    try:
        for index in range(1, count + 1):
            stem = f"ROIs1158_spring_s1_0_p{index}"
            sar_remote = f"ROIs1158_spring/s1_0/{stem}.png"
            opt_remote = f"ROIs1158_spring/s2_0/{stem.replace('_s1_', '_s2_')}.png"
            sample_id = f"sen12_{index:03d}"
            sar_local = output / "SEN1-2" / "sar" / f"{sample_id}.png"
            opt_local = output / "SEN1-2" / "opt" / f"{sample_id}.png"
            sar_local.parent.mkdir(parents=True, exist_ok=True)
            opt_local.parent.mkdir(parents=True, exist_ok=True)
            with sar_local.open("wb") as handle:
                ftp.retrbinary(f"RETR {sar_remote}", handle.write)
            with opt_local.open("wb") as handle:
                ftp.retrbinary(f"RETR {opt_remote}", handle.write)
            rows.append({
                "id": sample_id, "split": "train", "dataset": "SEN1-2",
                "opt_path": opt_local.relative_to(output).as_posix(),
                "sar_path": sar_local.relative_to(output).as_posix(),
                "opt_sensor": "Sentinel-2", "sar_sensor": "Sentinel-1",
                "opt_gsd": 10.0, "sar_gsd": 10.0, "polarization": "VV",
                "sar_product": "GRD", "sar_unit": "display_uint8",
                "source": "ftp://dataserv.ub.tum.de/FD_Server/m1436631",
            })
    finally:
        ftp.quit()
    return rows


def download_sar2opt(output, count):
    rows = []
    names = ["11_1200_0.jpg", "11_1200_240.jpg", "11_1200_480.jpg"][:count]
    for index, name in enumerate(names, 1):
        sar_source = hf_hub_download("umkc-mcc/SAR2Opt", f"A/test/{name}", repo_type="dataset")
        opt_source = hf_hub_download("umkc-mcc/SAR2Opt", f"B/test/{name}", repo_type="dataset")
        sample_id = f"sar2opt_{index:03d}"
        sar_local = output / "SAR2Opt" / "sar" / f"{sample_id}.jpg"
        opt_local = output / "SAR2Opt" / "opt" / f"{sample_id}.jpg"
        copy_image(sar_source, sar_local)
        copy_image(opt_source, opt_local)
        rows.append({
            "id": sample_id, "split": "train", "dataset": "SAR2Opt",
            "opt_path": opt_local.relative_to(output).as_posix(),
            "sar_path": sar_local.relative_to(output).as_posix(),
            "opt_sensor": "Google-Earth", "sar_sensor": "TerraSAR-X",
            "opt_gsd": 1.0, "sar_gsd": 1.0, "polarization": "unknown",
            "sar_product": "unknown", "sar_unit": "display_uint8",
            "source": "https://huggingface.co/datasets/umkc-mcc/SAR2Opt",
        })
    return rows


def import_folder(output, source_root, dataset, metadata, count):
    if not source_root:
        return []
    source_root = Path(source_root)
    opt_files = sorted((source_root / "opt").glob("*"))[:count]
    rows = []
    for index, opt_source in enumerate(opt_files, 1):
        sar_candidates = list((source_root / "sar").glob(f"{opt_source.stem}.*"))
        if not sar_candidates:
            continue
        sample_id = f"{dataset.lower().replace('-', '')}_{index:03d}"
        opt_local = output / dataset / "opt" / f"{sample_id}{opt_source.suffix.lower()}"
        sar_local = output / dataset / "sar" / f"{sample_id}{sar_candidates[0].suffix.lower()}"
        copy_image(opt_source, opt_local)
        copy_image(sar_candidates[0], sar_local)
        row = {
            "id": sample_id, "split": "train", "dataset": dataset,
            "opt_path": opt_local.relative_to(output).as_posix(),
            "sar_path": sar_local.relative_to(output).as_posix(),
            "source": str(source_root.resolve()), **metadata,
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/multidataset_smoke")
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--three-mos-root")
    parser.add_argument("--sar-one-m-root")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = download_sen12(output, args.count)
    rows += download_sar2opt(output, args.count)
    rows += import_folder(output, args.three_mos_root, "3MOS", {
        "opt_sensor": "Google-Earth", "sar_sensor": "unknown",
        "opt_gsd": 10.0, "sar_gsd": 10.0, "polarization": "unknown",
        "sar_unit": "display_uint8",
    }, args.count)
    rows += import_folder(output, args.sar_one_m_root, "SAR-1M", {
        "opt_sensor": "unknown", "sar_sensor": "unknown",
        "opt_gsd": 10.0, "sar_gsd": 10.0, "polarization": "unknown",
        "sar_unit": "display_uint8",
    }, args.count)
    manifest = output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    status = {
        "SEN1-2": "downloaded from official TUM FTP",
        "SAR2Opt": "downloaded from public Hugging Face mirror",
        "3MOS": "imported" if args.three_mos_root else "manual Baidu NetDisk access required",
        "SAR-1M": "imported" if args.sar_one_m_root else "official Hugging Face archive is gated and 76.6 GB",
    }
    (output / "SOURCE_STATUS.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(rows)} paired samples to {manifest}")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
