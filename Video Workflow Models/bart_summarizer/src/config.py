from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
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
    csv_path:    str   = "data_summarization.csv"
    article_col: str   = "lesson_text"
    summary_col: str   = "summary"
    test_size:   int   = 100
    val_ratio:   float = 0.1
    seed:        int   = 42


@dataclass
class ModelConfig:
    checkpoint:        str = "facebook/bart-large-cnn"
    max_input_length:  int = 1024
    max_target_length: int = 256


@dataclass
class LoraConfig:
    rank:           int       = 16
    alpha:          int       = 32
    dropout:        float     = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"
    ])


@dataclass
class TrainingConfig:
    epochs:                 int   = 8
    per_device_train_batch: int   = 4
    per_device_eval_batch:  int   = 4
    gradient_accum_steps:   int   = 4
    gradient_checkpointing: bool  = True
    warmup_steps:           int   = 100
    weight_decay:           float = 0.01
    logging_steps:          int   = 5
    save_total_limit:       int   = 3
    generation_max_length:  int   = 256
    seed:                   int   = 42
    fp16:                   bool  = True


@dataclass
class InferenceConfig:
    num_beams:  int = 4
    max_length: int = 256
    output_dir: str = "./outputs/inference"


@dataclass
class OutputConfig:
    dir:         str = "/vol/bart-lora-output"
    local_dir:   str = "./outputs/model"
    logging_dir: str = "/vol/logs"


@dataclass
class HubConfig:
    push_to_hub:    bool = False
    repo_id:        str  = ""
    private:        bool = False
    commit_message: str  = "Upload BART LoRA summarizer"


@dataclass
class ModalConfig:
    volume_name:    str = "led-summarizer-vol"
    gpu:            str = "H100"
    timeout:        int = 86400
    python_version: str = "3.11"
    torch_version:  str = "2.6.0"
    cuda_version:   str = "cu124"


@dataclass
class Config:
    credentials: Credentials   = field(default_factory=Credentials)
    dataset:     DatasetConfig  = field(default_factory=DatasetConfig)
    model:       ModelConfig    = field(default_factory=ModelConfig)
    lora:        LoraConfig     = field(default_factory=LoraConfig)
    training:    TrainingConfig = field(default_factory=TrainingConfig)
    inference:   InferenceConfig = field(default_factory=InferenceConfig)
    output:      OutputConfig   = field(default_factory=OutputConfig)
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
        ("dataset",   DatasetConfig,   "dataset"),
        ("model",     ModelConfig,     "model"),
        ("lora",      LoraConfig,      "lora"),
        ("training",  TrainingConfig,  "training"),
        ("inference", InferenceConfig, "inference"),
        ("output",    OutputConfig,    "output"),
        ("hub",       HubConfig,       "hub"),
        ("modal",     ModalConfig,     "modal"),
    ]:
        data = raw.get(section, {})
        setattr(cfg, attr, cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        }))

    return cfg
