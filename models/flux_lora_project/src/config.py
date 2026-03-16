from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

try:
    from dotenv import load_dotenv
    _env = Path.cwd() / ".env"
    if not _env.exists():
        _env = Path(__file__).parent.parent / ".env"
    if _env.exists():
        load_dotenv(_env)
        print(f"✅ Loaded .env from {_env}")
except ImportError:
    pass


@dataclass
class Credentials:
    hf_token:        str = ""
    kaggle_username: str = ""
    kaggle_key:      str = ""


@dataclass
class DatasetConfig:
    source:           str = "kaggle"
    kaggle_dataset:   str = ""
    hf_dataset:       str = ""
    local_path:       str = "./data"
    image_col:        str = "image"
    caption_col:      str = "description"
    image_extensions: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".webp", ".bmp",
        ".tiff", ".tif", ".heic", ".heif", ".avif", ".jfif"
    ])


@dataclass
class ModelConfig:
    name:     str = "black-forest-labs/FLUX.1-dev"
    revision: str = "main"


@dataclass
class LoraConfig:
    rank:           int       = 16
    alpha:          int       = 16
    dropout:        float     = 0.0
    bias:           str       = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0",
        "add_q_proj", "add_k_proj", "add_v_proj"
    ])


@dataclass
class TrainingConfig:
    resolution:           int   = 512
    batch_size:           int   = 1
    gradient_accum_steps: int   = 4
    max_steps:            int   = 1000
    learning_rate:        float = 1e-4
    lr_scheduler:         str   = "cosine"
    lr_warmup_steps:      int   = 100
    max_grad_norm:        float = 1.0
    seed:                 int   = 42
    guidance_scale:       float = 3.5
    mixed_precision:      str   = "bf16"
    num_workers:          int   = 2


@dataclass
class CheckpointingConfig:
    save_steps:   int = 200
    output_dir:   str = "/vol/flux-lora-output"
    local_output: str = "./outputs"


@dataclass
class InferenceConfig:
    prompt:              str   = "a portrait of a character"
    num_images:          int   = 4
    num_inference_steps: int   = 28
    guidance_scale:      float = 3.5
    lora_scale:          float = 0.9
    seed:                int   = 42
    output_dir:          str   = "/vol/inference_outputs"
    local_output:        str   = "./outputs/inference"


@dataclass
class HubConfig:
    push_to_hub:    bool = True
    repo_id:        str  = ""
    private:        bool = False
    commit_message: str  = "Upload FLUX LoRA fine-tuned model"


@dataclass
class ModalConfig:
    app_name:       str = "flux-lora"
    volume_name:    str = "flux-lora-vol"
    gpu:            str = "H100"
    timeout:        int = 14400
    python_version: str = "3.11"
    torch_version:  str = "2.5.1"
    cuda_version:   str = "cu124"


@dataclass
class Config:
    credentials:   Credentials      = field(default_factory=Credentials)
    dataset:       DatasetConfig     = field(default_factory=DatasetConfig)
    model:         ModelConfig       = field(default_factory=ModelConfig)
    lora:          LoraConfig        = field(default_factory=LoraConfig)
    training:      TrainingConfig    = field(default_factory=TrainingConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    inference:     InferenceConfig   = field(default_factory=InferenceConfig)
    hub:           HubConfig         = field(default_factory=HubConfig)
    modal:         ModalConfig       = field(default_factory=ModalConfig)


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load config from YAML file, override with environment variables."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    c = raw.get("credentials", {})
    cfg.credentials = Credentials(
        hf_token        = os.environ.get("HF_TOKEN",        c.get("hf_token", "")),
        kaggle_username = os.environ.get("KAGGLE_USERNAME", c.get("kaggle_username", "")),
        kaggle_key      = os.environ.get("KAGGLE_KEY",      c.get("kaggle_key", "")),
    )

    d = raw.get("dataset", {})
    cfg.dataset = DatasetConfig(**{k: v for k, v in d.items()
                                   if k in DatasetConfig.__dataclass_fields__})

    m = raw.get("model", {})
    cfg.model = ModelConfig(**{k: v for k, v in m.items()
                                if k in ModelConfig.__dataclass_fields__})

    l = raw.get("lora", {})
    cfg.lora = LoraConfig(**{k: v for k, v in l.items()
                              if k in LoraConfig.__dataclass_fields__})

    t = raw.get("training", {})
    cfg.training = TrainingConfig(**{k: v for k, v in t.items()
                                      if k in TrainingConfig.__dataclass_fields__})

    ck = raw.get("checkpointing", {})
    cfg.checkpointing = CheckpointingConfig(**{k: v for k, v in ck.items()
                                               if k in CheckpointingConfig.__dataclass_fields__})

    i = raw.get("inference", {})
    cfg.inference = InferenceConfig(**{k: v for k, v in i.items()
                                        if k in InferenceConfig.__dataclass_fields__})

    h = raw.get("hub", {})
    cfg.hub = HubConfig(**{k: v for k, v in h.items()
                            if k in HubConfig.__dataclass_fields__})

    mo = raw.get("modal", {})
    cfg.modal = ModalConfig(**{k: v for k, v in mo.items()
                                if k in ModalConfig.__dataclass_fields__})

    return cfg
