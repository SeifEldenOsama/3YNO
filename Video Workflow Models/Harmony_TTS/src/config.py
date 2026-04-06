from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
import yaml

try:
    from dotenv import load_dotenv
    _env = Path.cwd() / ".env"
    if not _env.exists():
        _env = Path(__file__).parent.parent / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass


@dataclass
class Credentials:
    hf_token: str = ""


@dataclass
class DatasetConfig:
    repo_id:           str = "SeifElden2342532/parler-tts-dataset-format"
    description_col:   str = "text_description"
    prompt_col:        str = "text"
    audio_col:         str = "audio"
    train_split:       str = "train"
    eval_split:        str = "validation"
    max_train_samples: int = 18700
    max_eval_samples:  int = 2000
    seed:              int = 42


@dataclass
class ModelConfig:
    name:                  str = "parler-tts/parler-tts-mini-v1"
    description_tokenizer: str = "google/flan-t5-base"
    sampling_rate:         int = 44100


@dataclass
class TrainingConfig:
    output_dir:             str   = "/vol/harmony-tts-output"
    local_output:           str   = "./outputs/model"
    per_device_train_batch: int   = 4
    per_device_eval_batch:  int   = 4
    gradient_accum_steps:   int   = 4
    learning_rate:          float = 1e-5
    lr_scheduler:           str   = "cosine"
    warmup_steps:           int   = 200
    max_steps:              int   = 2000
    eval_steps:             int   = 200
    save_steps:             int   = 200
    weight_decay:           float = 0.01
    bf16:                   bool  = True
    gradient_checkpointing: bool  = True
    seed:                   int   = 42


@dataclass
class HubConfig:
    repo_id:        str  = "SeifElden2342532/Harmony_Parler_TTS"
    private:        bool = False
    commit_message: str  = "Upload Harmony TTS full fine-tuned model"


@dataclass
class ModalConfig:
    app_name:       str = "harmony-tts"
    volume_name:    str = "tts-dataset-storage"
    gpu:            str = "H100:1"
    timeout:        int = 25000
    python_version: str = "3.11"


@dataclass
class Config:
    credentials: Credentials   = field(default_factory=Credentials)
    dataset:     DatasetConfig  = field(default_factory=DatasetConfig)
    model:       ModelConfig    = field(default_factory=ModelConfig)
    training:    TrainingConfig = field(default_factory=TrainingConfig)
    hub:         HubConfig      = field(default_factory=HubConfig)
    modal:       ModalConfig    = field(default_factory=ModalConfig)


def load_config(path: str = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    c = raw.get("credentials", {})
    cfg.credentials = Credentials(
        hf_token=os.environ.get("HF_TOKEN", c.get("hf_token", ""))
    )

    for section, cls, attr in [
        ("dataset",  DatasetConfig,  "dataset"),
        ("model",    ModelConfig,    "model"),
        ("training", TrainingConfig, "training"),
        ("hub",      HubConfig,      "hub"),
        ("modal",    ModalConfig,    "modal"),
    ]:
        data = raw.get(section, {})
        setattr(cfg, attr, cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        }))

    return cfg
