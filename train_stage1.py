"""Stage 1: large-scale paired OPT-to-SAR latent diffusion pretraining."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lc_osar.config import load_config
from lc_osar.data import MultiDatasetOptSarDataset, metadata_from_batch
from lc_osar.losses import AdaptivePhysicalWeight, sar_physical_loss
from lc_osar.models import Opt2SarLDM


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(cfg):
    data_cfg = cfg["data"]
    dataset = MultiDatasetOptSarDataset(
        data_cfg["manifest"], "train", data_cfg["image_size"], cfg["metadata"]
    )
    sampler = dataset.make_balanced_sampler(data_cfg.get("sampling_temperature", 0.5))
    return DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        sampler=sampler,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def build_validation_loader(cfg):
    data_cfg = cfg["data"]
    dataset = MultiDatasetOptSarDataset(
        data_cfg["manifest"], "val", data_cfg["image_size"], cfg["metadata"]
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("validation_num_samples", 4)),
        shuffle=False,
        num_workers=min(2, data_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )


def save_image(tensor, path):
    tensor = tensor.detach().float().clamp(-1, 1).add(1).mul(127.5).byte().cpu()
    array = tensor.numpy()
    if array.shape[0] == 1:
        Image.fromarray(array[0]).save(path)
    else:
        Image.fromarray(array[:3].transpose(1, 2, 0)).save(path)


@torch.inference_mode()
def generate_validation_samples(model, loader, cfg, device, epoch, output_dir, writer, use_amp):
    """Generate a fixed first validation batch for qualitative monitoring."""
    model.eval()
    batch = next(iter(loader))
    optical = batch["opt"].to(device, non_blocking=True)
    metadata = metadata_from_batch(batch, device)
    sample_steps = int(cfg["train"].get("validation_inference_steps", 50))
    scheduler = DDIMScheduler(
        num_train_timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        beta_schedule="linear", prediction_type="epsilon", clip_sample=False,
    )
    scheduler.set_timesteps(sample_steps, device=device)
    latent_size = cfg["data"]["image_size"] // 8
    generator = torch.Generator(device=device).manual_seed(int(cfg.get("seed", 42)))
    latent = torch.randn(
        len(optical), model.unet.config.in_channels, latent_size, latent_size,
        generator=generator, device=device,
    )
    for timestep in tqdm(scheduler.timesteps, desc=f"validation epoch {epoch}", leave=False):
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            noise = model(latent, timestep.expand(len(optical)), optical, metadata)
        latent = scheduler.step(noise.float(), timestep, latent).prev_sample
    generated = model.decode_sar(latent)
    sample_dir = output_dir / "samples" / f"epoch_{epoch:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(batch["stem"]):
        save_image(optical[index], sample_dir / f"{stem}_opt.png")
        save_image(batch["sar"][index], sample_dir / f"{stem}_real_sar.png")
        save_image(generated[index], sample_dir / f"{stem}_generated_sar.png")
    writer.add_images("validation/generated_sar", generated.add(1).div(2), epoch)
    writer.add_images("validation/real_sar", batch["sar"].add(1).div(2), epoch)
    writer.flush()
    model.train()
    model.vae.eval()
    print(f"[validation] saved {len(generated)} samples to {sample_dir}")


def save_checkpoint(path, model, optimizer, scaler, controller, epoch, global_step, cfg):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "physical_controller": controller.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": cfg,
    }, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", help="Override data.manifest from YAML")
    parser.add_argument("--output-dir", help="Override train.output_dir from YAML")
    parser.add_argument("--resume-from", help="Override train.resume_from from YAML")
    parser.add_argument("--batch-size", type=int, help="Override per-process data.batch_size")
    parser.add_argument("--epochs", type=int, help="Override total target epoch count")
    parser.add_argument("--lr", type=float, help="Override learning rate, including after resume")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.manifest:
        cfg["data"]["manifest"] = args.manifest
    if args.output_dir:
        cfg["train"]["output_dir"] = args.output_dir
    if args.resume_from:
        cfg["train"]["resume_from"] = args.resume_from
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Opt2SarLDM(cfg).to(device)
    model.vae.eval()
    scheduler = DDPMScheduler(
        num_train_timesteps=cfg["diffusion"]["timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        beta_schedule="linear",
        prediction_type="epsilon",
    )
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    use_amp = device.type == "cuda" and cfg["train"].get("mixed_precision") == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loader = build_loader(cfg)
    validation_loader = build_validation_loader(cfg)
    physical_controller = AdaptivePhysicalWeight(cfg["loss"])

    output_dir = Path(cfg["train"]["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    start_epoch = global_step = 0
    resume = cfg["train"].get("resume_from")
    if resume:
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if args.lr is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
                param_group["initial_lr"] = args.lr
        scaler.load_state_dict(state.get("scaler", {}))
        physical_controller.load_state_dict(state.get("physical_controller", {}))
        start_epoch = int(state["epoch"])
        global_step = int(state.get("global_step", 0))

    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    physical_interval = int(cfg["loss"].get("physical_interval", 1))
    physical_warmup = int(cfg["loss"].get("physical_warmup_steps", 0))
    physical_max_timestep = int(cfg["loss"].get("physical_max_timestep", cfg["diffusion"]["timesteps"] - 1))
    adaptive_update_interval = int(cfg["loss"].get("physical_update_interval", 50))
    log_every = int(cfg["train"].get("log_every_steps", 50))
    probe_parameter = model.unet.conv_out.weight
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        model.vae.eval()
        sums = {"loss": 0.0, "diffusion": 0.0, "physical": 0.0}
        steps = 0
        progress = tqdm(loader, desc=f"stage1 epoch {epoch + 1}")
        for batch_index, batch in enumerate(progress):
            max_steps = cfg["train"].get("max_steps")
            if max_steps is not None and batch_index >= max_steps:
                break
            opt = batch["opt"].to(device, non_blocking=True)
            sar = batch["sar"].to(device, non_blocking=True)
            metadata = metadata_from_batch(batch, device)
            with torch.no_grad():
                clean_latent = model.encode_sar(sar)
            noise = torch.randn_like(clean_latent)
            timestep = torch.randint(0, scheduler.config.num_train_timesteps, (len(sar),), device=device)
            noisy_latent = scheduler.add_noise(clean_latent, noise, timestep)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(noisy_latent, timestep, opt, metadata)
                diffusion_loss = F.mse_loss(prediction.float(), noise.float())
                physical_loss = diffusion_loss.new_zeros(())
                physical_metrics = {}
                physical_mask = timestep <= physical_max_timestep
                use_physical = (
                    physical_interval > 0
                    and global_step >= physical_warmup
                    and global_step % physical_interval == 0
                    and bool(physical_mask.any())
                )
                if use_physical:
                    selected_t = timestep[physical_mask]
                    alpha = alphas_cumprod[selected_t].view(-1, 1, 1, 1)
                    pred_clean = (
                        noisy_latent[physical_mask]
                        - (1.0 - alpha).sqrt() * prediction[physical_mask]
                    ) / alpha.sqrt()
                    pred_sar = model.decode_sar(pred_clean, with_grad=True)
                    physical_loss, physical_metrics = sar_physical_loss(
                        pred_sar.float(), sar[physical_mask].float(),
                        metadata["sar_gsd"][physical_mask], cfg["loss"]
                    )
                    if (
                        physical_controller.enabled
                        and adaptive_update_interval > 0
                        and global_step % adaptive_update_interval == 0
                    ):
                        physical_controller.update(diffusion_loss, physical_loss, probe_parameter)
                lambda_physical = physical_controller.value if use_physical else 0.0
                loss = (
                    cfg["loss"].get("lambda_diffusion", 1.0) * diffusion_loss
                    + lambda_physical * physical_loss
                )

            scaler.scale(loss / grad_accum).backward()
            if (batch_index + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, cfg["train"].get("grad_clip", 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            values = {
                "loss": loss, "diffusion": diffusion_loss, "physical": physical_loss,
                "lambda_physical": loss.new_tensor(lambda_physical), **physical_metrics,
            }
            for key, value in values.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach())
            steps += 1
            global_step += 1
            progress.set_postfix(
                loss=f"{sums['loss'] / steps:.4f}",
                diffusion=f"{float(diffusion_loss.detach()):.4f}",
                physical=f"{float(physical_loss.detach()):.4f}",
                lambda_phy=f"{lambda_physical:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            if log_every > 0 and global_step % log_every == 0:
                for key, value in values.items():
                    writer.add_scalar(f"train/{key}", float(value.detach()), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        if steps % grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, cfg["train"].get("grad_clip", 1.0))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        row = {"epoch": epoch + 1, "global_step": global_step, **{k: v / max(steps, 1) for k, v in sums.items()}}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        for key, value in row.items():
            if key not in {"epoch", "global_step"}:
                writer.add_scalar(f"epoch/{key}", value, epoch + 1)
        writer.flush()
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, scaler, physical_controller, epoch + 1, global_step, cfg)
        if (epoch + 1) % cfg["train"].get("save_every_epochs", 1) == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1:04d}.pt", model, optimizer, scaler, physical_controller, epoch + 1, global_step, cfg)
        validation_interval = int(cfg["train"].get("validation_every_epochs", 0))
        if validation_interval > 0 and (epoch + 1) % validation_interval == 0:
            generate_validation_samples(
                model, validation_loader, cfg, device, epoch + 1,
                output_dir, writer, use_amp,
            )
        print(row)
    writer.close()


if __name__ == "__main__":
    main()
