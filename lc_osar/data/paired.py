"""Manifest-based multi-dataset OPT-SAR loading."""

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
UNKNOWN = "unknown"


class DistributedWeightedSampler(Sampler):
    """Deterministic weighted global sampling followed by rank-wise sharding."""

    def __init__(self, weights, num_replicas, rank, seed=42):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = math.ceil(len(self.weights) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=generator
        ).tolist()
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def _read_manifest(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError("Manifest must be .jsonl or .csv")


def _resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def _image_tensor(path, mode, size):
    image = Image.open(path).convert(mode).resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image).copy()
    if mode == "L":
        array = array[None]
    else:
        array = array.transpose(2, 0, 1)
    return torch.from_numpy(array).float().div(127.5).sub(1.0)


def _vocab_index(value, values):
    value = str(value or UNKNOWN)
    mapping = {name: index for index, name in enumerate(values)}
    return mapping.get(value, mapping.get(UNKNOWN, 0))


class MultiDatasetOptSarDataset(Dataset):
    """Paired samples plus physical acquisition metadata.

    Required manifest fields: id, split, opt_path, sar_path.
    Recommended: dataset, opt_sensor, sar_sensor, sar_gsd,
    opt_gsd, polarization, incidence_angle.
    """

    def __init__(self, manifest, split, image_size, metadata_cfg, require_sar=True,
                 validate_paths=True):
        self.manifest_path = Path(manifest)
        self.base = self.manifest_path.parent
        self.image_size = int(image_size)
        self.metadata_cfg = metadata_cfg
        records = _read_manifest(self.manifest_path)
        self.records = [row for row in records if not split or row.get("split", "train") == split]
        if not self.records:
            raise RuntimeError(f"No '{split}' samples in {self.manifest_path}")
        if validate_paths:
            for row in self.records:
                opt_path = _resolve(self.base, row["opt_path"])
                sar_path = _resolve(self.base, row["sar_path"]) if row.get("sar_path") else None
                if not opt_path.is_file():
                    raise FileNotFoundError(f"Missing OPT image: {opt_path}")
                if require_sar and (sar_path is None or not sar_path.is_file()):
                    raise FileNotFoundError(f"Missing SAR image for {row.get('id')}: {sar_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        opt_path = _resolve(self.base, row["opt_path"])
        sar_path = _resolve(self.base, row["sar_path"]) if row.get("sar_path") else None
        gsd_default = float(self.metadata_cfg.get("default_gsd", 10.0))
        item = {
            "opt": _image_tensor(opt_path, "RGB", self.image_size),
            "stem": str(row.get("id", opt_path.stem)),
            "dataset_name": str(row.get("dataset", UNKNOWN)),
            "dataset_id": _vocab_index(row.get("dataset"), self.metadata_cfg["datasets"]),
            "opt_sensor_id": _vocab_index(row.get("opt_sensor"), self.metadata_cfg["opt_sensors"]),
            "sar_sensor_id": _vocab_index(row.get("sar_sensor"), self.metadata_cfg["sar_sensors"]),
            "polarization_id": _vocab_index(row.get("polarization"), self.metadata_cfg["polarizations"]),
            "opt_gsd": float(row.get("opt_gsd") or gsd_default),
            "sar_gsd": float(row.get("sar_gsd") or gsd_default),
            "incidence_angle": float(row.get("incidence_angle") or 0.0),
            "metadata_known": float(bool(row.get("sar_gsd") or row.get("sar_sensor"))),
        }
        if sar_path is not None and sar_path.is_file():
            item["sar"] = _image_tensor(sar_path, "L", self.image_size)
        return item

    def make_balanced_sampler(self, temperature=0.5, replacement=True):
        """Sample datasets with p(dataset) proportional to N**temperature."""
        counts = Counter(str(row.get("dataset", UNKNOWN)) for row in self.records)
        weights = [counts[str(row.get("dataset", UNKNOWN))] ** (float(temperature) - 1.0) for row in self.records]
        return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=replacement)

    def make_distributed_balanced_sampler(self, num_replicas, rank, temperature=0.5, seed=42):
        counts = Counter(str(row.get("dataset", UNKNOWN)) for row in self.records)
        weights = [counts[str(row.get("dataset", UNKNOWN))] ** (float(temperature) - 1.0) for row in self.records]
        return DistributedWeightedSampler(weights, num_replicas, rank, seed)


def metadata_from_batch(batch, device, dtype=None):
    float_dtype = dtype or torch.float32
    return {
        "dataset_id": batch["dataset_id"].to(device),
        "opt_sensor_id": batch["opt_sensor_id"].to(device),
        "sar_sensor_id": batch["sar_sensor_id"].to(device),
        "polarization_id": batch["polarization_id"].to(device),
        "opt_gsd": batch["opt_gsd"].to(device=device, dtype=float_dtype),
        "sar_gsd": batch["sar_gsd"].to(device=device, dtype=float_dtype),
        "incidence_angle": batch["incidence_angle"].to(device=device, dtype=float_dtype),
        "metadata_known": batch["metadata_known"].to(device=device, dtype=float_dtype),
    }
