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
class ModelConfig:
    id:             str   = "Qwen/Qwen2.5-32B-Instruct"
    cache_dir:      str   = "/model-cache"
    max_new_tokens: int   = 4000
    temperature:    float = 0.7
    top_p:          float = 0.9


@dataclass
class StoryConfig:
    min_characters:  int = 2
    max_characters:  int = 6
    min_backgrounds: int = 2
    max_backgrounds: int = 6
    min_scenes:      int = 3
    max_scenes:      int = 10


@dataclass
class OutputConfig:
    dir: str = "outputs"


@dataclass
class ModalConfig:
    app_name:       str = "kids-story-generator"
    volume_name:    str = "story-model-cache"
    gpu:            str = "A100"
    timeout:        int = 3600
    python_version: str = "3.11"


@dataclass
class Config:
    credentials: Credentials  = field(default_factory=Credentials)
    model:       ModelConfig   = field(default_factory=ModelConfig)
    story:       StoryConfig   = field(default_factory=StoryConfig)
    output:      OutputConfig  = field(default_factory=OutputConfig)
    modal:       ModalConfig   = field(default_factory=ModalConfig)


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
        ("model",  ModelConfig,  "model"),
        ("story",  StoryConfig,  "story"),
        ("output", OutputConfig, "output"),
        ("modal",  ModalConfig,  "modal"),
    ]:
        data = raw.get(section, {})
        setattr(cfg, attr, cls(**{
            k: v for k, v in data.items()
            if k in cls.__dataclass_fields__
        }))

    return cfg
