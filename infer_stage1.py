"""Generate SAR images from optical images with a Stage 1 checkpoint."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from lc_osar.config import load_config
from lc_osar.data import MultiDatasetOptSarDataset, metadata_from_batch
from lc_osar.models import Opt2SarLDM


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_grayscale(tensor, path):
    array = tensor.detach().float().clamp(-1, 1).add(1).mul(127.5).byte().cpu().numpy()
    Image.fromarray(array[0]).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="Inference .jsonl/.csv manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    args = parser.parse_args()
    seed_everything(args.seed)
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if args.precision == "fp16" and device.type == "cuda" else torch.float32

    dataset = MultiDatasetOptSarDataset(
        args.manifest, split=None, image_size=cfg["data"]["image_size"],
        metadata_cfg=cfg["metadata"], require_sar=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=cfg["data"].get("num_workers", 4))
    model = Opt2SarLDM(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False).eval().to(device=device, dtype=dtype)
    scheduler = DDIMScheduler(
        num_train_timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )
    scheduler.set_timesteps(args.steps, device=device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    latent_size = cfg["data"]["image_size"] // 8

    with torch.inference_mode():
        for batch in tqdm(loader, desc="OPT to SAR"):
            optical = batch["opt"].to(device=device, dtype=dtype)
            metadata = metadata_from_batch(batch, device, dtype=dtype)
            latent = torch.randn(len(optical), model.unet.config.in_channels, latent_size, latent_size, device=device, dtype=dtype)
            for timestep in scheduler.timesteps:
                timestep_batch = timestep.expand(len(optical))
                noise = model(latent, timestep_batch, optical, metadata)
                latent = scheduler.step(noise, timestep, latent).prev_sample
            generated = model.decode_sar(latent)
            for image, stem in zip(generated, batch["stem"]):
                save_grayscale(image, output / f"{stem}.png")


if __name__ == "__main__":
    main()
