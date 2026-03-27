from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    id:        str
    pipeline:  str
    lora_id:   str
    lora_scale: float = 0.8
    cache_dir: str    = "/model-cache"


@dataclass
class GenerationConfig:
    fps:            float = 24.0
    num_steps:      int   = 8
    guidance_scale: float = 1.0
    max_duration:   float = 12.0
    default_prompt: str   = (
        "A cute animated water droplet character with a round expressive face, "
        "big glossy eyes blinking naturally, mouth and lips moving clearly in sync with speech, "
        "subtle bouncing and swaying motion, surrounded by a vibrant underwater ocean scene "
        "with colorful tropical fish swimming around, soft blue-green water caustics lighting, "
        "coral reef in background, bubbles rising, smooth fluid animation, high quality"
    )


@dataclass
class Config:
    model:           ModelConfig
    generation:      GenerationConfig
    negative_prompt: str = "low quality, worst quality, deformed, distorted, static, frozen, no movement"


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    model = ModelConfig(
        id        = raw["model"]["id"],
        pipeline  = raw["model"]["pipeline"],
        lora_id   = raw["model"]["lora_id"],
        lora_scale= raw["model"].get("lora_scale", 0.8),
        cache_dir = raw["model"].get("cache_dir", "/model-cache"),
    )

    g_raw = raw.get("generation", {})
    generation = GenerationConfig(
        fps            = g_raw.get("fps",            24.0),
        num_steps      = g_raw.get("num_steps",      8),
        guidance_scale = g_raw.get("guidance_scale", 1.0),
        max_duration   = g_raw.get("max_duration",   12.0),
        default_prompt = g_raw.get("default_prompt", GenerationConfig.default_prompt),
    )

    return Config(
        model           = model,
        generation      = generation,
        negative_prompt = raw.get("negative_prompt", Config.negative_prompt),
    )
