from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import Config


def build_transforms(resolution: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])


def download_kaggle_dataset(cfg: Config, data_dir: str = "/tmp/kaggle_data") -> str:
    """Download and unzip a Kaggle dataset. Returns the data directory path."""
    os.environ["KAGGLE_USERNAME"] = cfg.credentials.kaggle_username
    os.environ["KAGGLE_KEY"]      = cfg.credentials.kaggle_key

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset: {cfg.dataset.kaggle_dataset} ...")

    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", cfg.dataset.kaggle_dataset,
            "-p", data_dir,
            "--unzip",
        ],
        check=True,
    )
    print(f"Dataset downloaded to {data_dir}")
    return data_dir


def collect_image_paths(data_dir: str, extensions: List[str]) -> List[str]:
    """Walk a directory and collect all image file paths."""
    ext_set = set(e.lower() for e in extensions)
    paths: List[str] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if Path(f).suffix.lower() in ext_set:
                paths.append(os.path.join(root, f))
    paths.sort()
    return paths


class KaggleDataset(Dataset):
    def __init__(self, image_paths: List[str], transform: transforms.Compose):
        self.paths     = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.paths[idx]
        txt_path = str(Path(img_path).with_suffix(".txt"))

        img = Image.open(img_path).convert("RGB")

        caption = "a character"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip() or "a character"

        return self.transform(img), caption


class HFDataset(Dataset):
    """Wraps a HuggingFace dataset split."""

    def __init__(self, hf_dataset, image_col: str, caption_col: str,
                 transform: transforms.Compose):
        self.data        = hf_dataset
        self.image_col   = image_col
        self.caption_col = caption_col
        self.transform   = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        item    = self.data[idx]
        img     = item[self.image_col]
        if not isinstance(img, Image.Image):
            import numpy as np
            img = Image.fromarray(img)
        img     = img.convert("RGB")
        caption = str(item[self.caption_col])
        return self.transform(img), caption

class LocalDataset(Dataset):
    """Same as KaggleDataset but for a pre-existing local folder."""

    def __init__(self, image_paths: List[str], transform: transforms.Compose):
        self.paths     = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.paths[idx]
        txt_path = str(Path(img_path).with_suffix(".txt"))

        img     = Image.open(img_path).convert("RGB")
        caption = "a character"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip() or "a character"

        return self.transform(img), caption


def build_dataset(cfg: Config) -> Dataset:
    """Build the correct dataset based on config.dataset.source."""
    transform = build_transforms(cfg.training.resolution)
    source    = cfg.dataset.source.lower()

    if source == "kaggle":
        data_dir    = download_kaggle_dataset(cfg)
        image_paths = collect_image_paths(data_dir, cfg.dataset.image_extensions)
        print(f"Found {len(image_paths)} images")
        return KaggleDataset(image_paths, transform)

    elif source == "huggingface":
        from datasets import load_dataset as hf_load
        print(f"Loading HuggingFace dataset: {cfg.dataset.hf_dataset} ...")
        raw = hf_load(cfg.dataset.hf_dataset, split="train")
        print(f"Dataset loaded: {len(raw)} samples")
        return HFDataset(raw, cfg.dataset.image_col, cfg.dataset.caption_col, transform)

    elif source == "local":
        image_paths = collect_image_paths(cfg.dataset.local_path, cfg.dataset.image_extensions)
        print(f"Found {len(image_paths)} local images")
        return LocalDataset(image_paths, transform)

    else:
        raise ValueError(f"Unknown dataset source: '{source}'. Choose: kaggle | huggingface | local")
