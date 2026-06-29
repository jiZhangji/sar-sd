"""Stage 1 conditional latent diffusion model."""

import warnings

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from torch import nn


def _load_unet(path, model_cfg):
    if model_cfg.get("unet_from_scratch", False):
        return UNet2DConditionModel(
            sample_size=model_cfg.get("sample_size", 8),
            in_channels=4,
            out_channels=4,
            layers_per_block=model_cfg.get("layers_per_block", 1),
            block_out_channels=tuple(model_cfg.get("block_out_channels", [32, 64])),
            down_block_types=tuple(model_cfg.get("down_block_types", ["CrossAttnDownBlock2D", "DownBlock2D"])),
            up_block_types=tuple(model_cfg.get("up_block_types", ["UpBlock2D", "CrossAttnUpBlock2D"])),
            cross_attention_dim=model_cfg.get("cross_attention_dim", 64),
            attention_head_dim=model_cfg.get("attention_head_dim", 8),
            norm_num_groups=model_cfg.get("norm_num_groups", 8),
        )
    try:
        return UNet2DConditionModel.from_pretrained(path, subfolder="unet")
    except (OSError, ValueError):
        try:
            return UNet2DConditionModel.from_pretrained(path)
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Cannot load UNet weights from {path}") from error


class OpticalConditionEncoder(nn.Module):
    """Turn an RGB optical image into cross-attention tokens."""

    def __init__(self, feature_dim=384, cross_attention_dim=768, token_grid=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(128, feature_dim, 3, 2, 1), nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((token_grid, token_grid))
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim),
        )

    def forward(self, optical):
        features = self.pool(self.encoder(optical)).flatten(2).transpose(1, 2)
        return self.projection(features)


class AcquisitionMetadataEncoder(nn.Module):
    """Encode physical acquisition metadata as one cross-attention token."""

    def __init__(self, metadata_cfg, output_dim, dropout=0.1):
        super().__init__()
        embedding_dim = int(metadata_cfg.get("embedding_dim", 64))
        self.dataset = nn.Embedding(len(metadata_cfg["datasets"]), embedding_dim)
        self.opt_sensor = nn.Embedding(len(metadata_cfg["opt_sensors"]), embedding_dim)
        self.sar_sensor = nn.Embedding(len(metadata_cfg["sar_sensors"]), embedding_dim)
        self.polarization = nn.Embedding(len(metadata_cfg["polarizations"]), embedding_dim)
        self.dataset_dropout = float(metadata_cfg.get("dataset_dropout", dropout))
        continuous_dim = 18  # 8 Fourier dims per GSD + angle + known flag
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * 4 + continuous_dim, output_dim),
            nn.SiLU(),
            nn.LayerNorm(output_dim),
        )

    @staticmethod
    def _fourier(value):
        value = torch.log(value.clamp_min(1e-3)).unsqueeze(1)
        frequencies = value.new_tensor([1.0, 2.0, 4.0, 8.0])
        angles = value * frequencies
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    def forward(self, metadata):
        dataset_id = metadata["dataset_id"]
        if self.training and self.dataset_dropout > 0:
            drop = torch.rand_like(dataset_id.float()) < self.dataset_dropout
            dataset_id = torch.where(drop, torch.zeros_like(dataset_id), dataset_id)
        categorical = torch.cat([
            self.dataset(dataset_id),
            self.opt_sensor(metadata["opt_sensor_id"]),
            self.sar_sensor(metadata["sar_sensor_id"]),
            self.polarization(metadata["polarization_id"]),
        ], dim=1)
        continuous = torch.cat([
            self._fourier(metadata["opt_gsd"]),
            self._fourier(metadata["sar_gsd"]),
            (metadata["incidence_angle"] / 90.0).unsqueeze(1),
            metadata["metadata_known"].unsqueeze(1),
        ], dim=1)
        return self.projection(torch.cat([categorical, continuous], dim=1)).unsqueeze(1)


class Opt2SarLDM(nn.Module):
    """Frozen SAR VAE + OPT condition encoder + conditional denoising UNet."""

    def __init__(self, cfg):
        super().__init__()
        model_cfg = cfg["model"]
        self.latent_scale = float(model_cfg.get("latent_scale", 0.18215))
        self.vae = AutoencoderKL.from_pretrained(model_cfg["vae_path"])
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.unet = _load_unet(model_cfg.get("unet_path"), model_cfg)
        if model_cfg.get("enable_gradient_checkpointing", False):
            self.unet.enable_gradient_checkpointing()
        if model_cfg.get("freeze_unet", False):
            self.unet.requires_grad_(False)
            warnings.warn("UNet is frozen; only the optical condition encoder will be trained.")
        self.opt_encoder = OpticalConditionEncoder(
            feature_dim=model_cfg.get("condition_dim", 384),
            cross_attention_dim=model_cfg.get("cross_attention_dim", self.unet.config.cross_attention_dim),
            token_grid=model_cfg.get("opt_token_grid", 8),
        )
        self.metadata_encoder = AcquisitionMetadataEncoder(
            cfg["metadata"],
            output_dim=model_cfg.get("cross_attention_dim", self.unet.config.cross_attention_dim),
        )

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        return self

    @torch.no_grad()
    def encode_sar(self, sar):
        rgb_sar = sar.repeat(1, 3, 1, 1)
        return self.vae.encode(rgb_sar).latent_dist.sample() * self.latent_scale

    def decode_sar(self, latent, with_grad=False):
        context = torch.enable_grad() if with_grad else torch.no_grad()
        with context:
            rgb = self.vae.decode(latent / self.latent_scale).sample
        return rgb.mean(dim=1, keepdim=True).clamp(-1.0, 1.0)

    def forward(self, noisy_latent, timestep, optical, metadata):
        optical_condition = self.opt_encoder(optical)
        metadata_condition = self.metadata_encoder(metadata)
        condition = torch.cat([optical_condition, metadata_condition], dim=1)
        return self.unet(noisy_latent, timestep, encoder_hidden_states=condition).sample
