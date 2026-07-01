"""Check whether the frozen SD VAE preserves SAR image structure."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lc_osar.config import load_config
from lc_osar.data import MultiDatasetOptSarDataset
from lc_osar.models import Opt2SarLDM


def to_uint8(tensor):
    array = tensor.detach().float().clamp(-1, 1).add(1).mul(127.5).byte().cpu().numpy()
    if array.shape[0] == 1:
        return np.repeat(array[0, :, :, None], 3, axis=2)
    return array[:3].transpose(1, 2, 0)


def save_panel(sar, recon, path):
    sar_rgb = to_uint8(sar)
    recon_rgb = to_uint8(recon)
    diff = np.abs(recon_rgb.astype(np.int16) - sar_rgb.astype(np.int16)).clip(0, 255).astype(np.uint8)
    panel = np.concatenate([sar_rgb, recon_rgb, diff], axis=1)
    Image.fromarray(panel).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", help="Override data.manifest from YAML")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.manifest:
        cfg["data"]["manifest"] = args.manifest
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = MultiDatasetOptSarDataset(
        cfg["data"]["manifest"], args.split, cfg["data"]["image_size"], cfg["metadata"],
        validate_paths=cfg["data"].get("validate_paths", False),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = Opt2SarLDM(cfg).to(device).eval()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    saved = 0
    total_l1 = 0.0
    total_mse = 0.0
    total_pixels = 0
    with torch.inference_mode():
        for batch in loader:
            sar = batch["sar"].to(device)
            recon = model.decode_sar(model.encode_sar(sar))
            total_l1 += float(F.l1_loss(recon, sar, reduction="sum").cpu())
            total_mse += float(F.mse_loss(recon, sar, reduction="sum").cpu())
            total_pixels += int(sar.numel())
            for index, stem in enumerate(batch["stem"]):
                save_panel(sar[index], recon[index], output / f"{stem}_sar_vae_recon.png")
                saved += 1
                if saved >= args.max_samples:
                    metrics = {
                        "split": args.split,
                        "samples": saved,
                        "mean_l1": total_l1 / max(total_pixels, 1),
                        "mean_mse": total_mse / max(total_pixels, 1),
                    }
                    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                    print(json.dumps(metrics, indent=2))
                    return

    metrics = {
        "split": args.split,
        "samples": saved,
        "mean_l1": total_l1 / max(total_pixels, 1),
        "mean_mse": total_mse / max(total_pixels, 1),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
