"""Download public OPT-SAR datasets from verified open sources.

Examples:
  python tools/download_datasets.py list
  python tools/download_datasets.py download sar2opt --output D:/datasets --limit 2
  python tools/download_datasets.py download sen1_2 --output D:/datasets --limit 2
  python tools/download_datasets.py download whu_opt_sar --output D:/datasets --full
  python tools/download_datasets.py download sar1m --output D:/datasets --full --token $env:HF_TOKEN
  python tools/download_datasets.py modelscope namespace/dataset --output D:/datasets/name
"""

import argparse
import json
import shutil
import subprocess
import sys
from ftplib import FTP
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "dataset_sources.yaml"


def load_registry():
    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["datasets"]


def copy_cached(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def download_sar2opt(spec, output, limit, token):
    limit = limit or 2
    api = HfApi(token=token)
    files = api.list_repo_files(spec["repo_id"], repo_type="dataset")
    a_files = sorted(name for name in files if name.startswith("A/train/") or name.startswith("A/test/"))
    selected = a_files[:limit]
    if not selected:
        raise RuntimeError("No SAR2Opt A/ files found")
    for sar_name in selected:
        opt_name = "B/" + sar_name[2:]
        sar_source = hf_hub_download(spec["repo_id"], sar_name, repo_type="dataset", token=token)
        opt_source = hf_hub_download(spec["repo_id"], opt_name, repo_type="dataset", token=token)
        copy_cached(sar_source, output / "sar" / Path(sar_name).name)
        copy_cached(opt_source, output / "opt" / Path(opt_name).name)
    print(f"downloaded {len(selected)} SAR2Opt pairs to {output}")


def download_sen12(output, limit):
    limit = limit or 2
    ftp = FTP("dataserv.ub.tum.de", timeout=90)
    ftp.login("m1436631", "m1436631")
    try:
        for index in range(1, limit + 1):
            sar_name = f"ROIs1158_spring_s1_0_p{index}.png"
            opt_name = sar_name.replace("_s1_", "_s2_")
            targets = [
                (f"ROIs1158_spring/s1_0/{sar_name}", output / "sar" / f"sen12_{index:04d}.png"),
                (f"ROIs1158_spring/s2_0/{opt_name}", output / "opt" / f"sen12_{index:04d}.png"),
            ]
            for remote, local in targets:
                local.parent.mkdir(parents=True, exist_ok=True)
                with local.open("wb") as handle:
                    ftp.retrbinary(f"RETR {remote}", handle.write)
    finally:
        ftp.quit()
    print(f"downloaded {limit} official SEN1-2 pairs to {output}")


def download_hf_archive(spec, output, token, full):
    size = spec.get("approximate_size_gb", "unknown")
    if not full:
        raise SystemExit(
            f"{spec['repo_id']} is archive-only (~{size} GB). "
            "Re-run with --full after checking disk space and license."
        )
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec["repo_id"], repo_type="dataset", token=token,
        allow_patterns=spec.get("include"), local_dir=output,
    )
    print(f"downloaded archives to {output}")


def download_google_drive(spec, output, full):
    if not full:
        raise SystemExit("SOMA-1M test is an archive. Re-run with --full after checking disk space.")
    try:
        import gdown
    except ImportError as error:
        raise SystemExit("Install optional dependency first: pip install gdown") from error
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "SOMA-1M-test.download"
    result = gdown.download(id=spec["file_id"], output=str(destination), quiet=False)
    if not result:
        raise RuntimeError("Google Drive download failed; use the official Baidu link if quota is exceeded")
    print(f"downloaded SOMA-1M test archive to {result}")


def download_registered(name, args):
    registry = load_registry()
    if name not in registry:
        raise SystemExit(f"Unknown dataset '{name}'. Run the list command.")
    spec = registry[name]
    output = Path(args.output).expanduser().resolve() / name
    provider = spec["provider"]
    if provider == "huggingface_files":
        download_sar2opt(spec, output, args.limit, args.token)
    elif provider == "tum_ftp":
        download_sen12(output, args.limit)
    elif provider == "huggingface_archive":
        download_hf_archive(spec, output, args.token, args.full)
    elif provider == "google_drive":
        download_google_drive(spec, output, args.full)
    elif provider == "application":
        output.mkdir(parents=True, exist_ok=True)
        note = {"dataset": name, "status": "manual access required", **spec}
        (output / "DOWNLOAD_REQUIRED.json").write_text(
            json.dumps(note, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"No verified direct archive. Follow {spec['url']}")
    else:
        raise RuntimeError(f"Unsupported provider: {provider}")


def download_modelscope(repo_id, output, include, token):
    command = ["modelscope", "download", "--dataset", repo_id, "--local_dir", str(Path(output).resolve())]
    for pattern in include or []:
        command.extend(["--include", pattern])
    if token:
        command.extend(["--token", token])
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise SystemExit("Install ModelScope CLI first: pip install modelscope") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    download = sub.add_parser("download")
    download.add_argument("name")
    download.add_argument("--output", required=True)
    download.add_argument("--limit", type=int)
    download.add_argument("--token")
    download.add_argument("--full", action="store_true")
    ms = sub.add_parser("modelscope")
    ms.add_argument("repo_id")
    ms.add_argument("--output", required=True)
    ms.add_argument("--include", action="append")
    ms.add_argument("--token")
    args = parser.parse_args()
    if args.command == "list":
        for name, spec in load_registry().items():
            print(f"{name:16s} {spec['provider']:20s} {spec['description']}")
    elif args.command == "download":
        download_registered(args.name, args)
    else:
        download_modelscope(args.repo_id, args.output, args.include, args.token)


if __name__ == "__main__":
    main()
