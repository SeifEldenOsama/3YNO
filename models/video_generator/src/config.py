from __future__ import annotations
from dataclasses import dataclass
import yaml


@dataclass
class ModelConfig:
    id:         str
    pipeline:   str
    lora_id:    str
    lora_scale: float = 0.8
    cache_dir:  str   = "/model-cache"


@dataclass
class GenerationConfig:
    fps:            float = 24.0
    num_steps:      int   = 8
    guidance_scale: float = 1.0
    max_duration:   float = 12.0
    quality:        str   = "fhd"
    default_prompt: str   = (
        "A cartoon character speaking expressively, "
        "mouth and lips moving clearly in sync with speech, "
        "smooth fluid animation, high quality"
    )


@dataclass
class Config:
    model:           ModelConfig
    generation:      GenerationConfig
    negative_prompt: str = (
        "low quality, worst quality, deformed, distorted, blurry, noisy, "
        "text, subtitles, captions, watermark, words, letters, typography, "
        "moving background, camera movement, panning, zooming, shaking, parallax, "
        "both characters talking simultaneously"
    )


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    model = ModelConfig(
        id         = raw["model"]["id"],
        pipeline   = raw["model"]["pipeline"],
        lora_id    = raw["model"]["lora_id"],
        lora_scale = raw["model"].get("lora_scale", 0.8),
        cache_dir  = raw["model"].get("cache_dir", "/model-cache"),
    )

    g = raw.get("generation", {})
    generation = GenerationConfig(
        fps            = g.get("fps",            24.0),
        num_steps      = g.get("num_steps",      8),
        guidance_scale = g.get("guidance_scale", 1.0),
        max_duration   = g.get("max_duration",   12.0),
        quality        = g.get("quality",        "fhd"),
        default_prompt = g.get("default_prompt", GenerationConfig.default_prompt),
    )

    return Config(
        model           = model,
        generation      = generation,
        negative_prompt = raw.get("negative_prompt", Config.negative_prompt),
    )