import yaml
import os
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ModelConfig:
    name: str = "zai-org/GLM-OCR"
    device: str = "cuda"
    dtype: str = "auto"

@dataclass
class InferenceConfig:
    max_new_tokens: int = 8192
    prompt: str = "Text Recognition:"
    local_output: str = "./outputs"

@dataclass
class ModalConfig:
    gpu: str = "A10G"
    timeout: int = 3600
    volume_name: str = "glm-ocr-cache"
    cache_dir: str = "/model-cache"

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    modal: ModalConfig = field(default_factory=ModalConfig)

    @classmethod
    def load(cls, path: str = None) -> "Config":
        if path is None:
            # Try to find config.yaml in the project root relative to this file
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(base_path, "config.yaml")
            
        if not os.path.exists(path):
            print(f"Warning: Config file not found at {path}. Using defaults.")
            return cls()
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**data.get("model", {})),
            inference=InferenceConfig(**data.get("inference", {})),
            modal=ModalConfig(**data.get("modal", {})),
        )
