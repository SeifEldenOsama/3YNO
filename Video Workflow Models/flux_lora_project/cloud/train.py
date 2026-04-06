import modal

VOLUME_NAME    = "flux-lora-vol"
GPU            = "H100"
TIMEOUT        = 14400
PYTHON_VERSION = "3.11"
TORCH_VERSION  = "2.5.1"
CUDA_VERSION   = "cu124"
OUTPUT_DIR     = "/vol/flux-lora-output"

import os
from dotenv import load_dotenv
load_dotenv(".env")

HF_TOKEN        = os.getenv("HF_TOKEN", "")
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY      = os.getenv("KAGGLE_KEY", "")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "libgl1", "libglib2.0-0", "unzip")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        "torchvision",
        index_url=f"https://download.pytorch.org/whl/{CUDA_VERSION}",
    )
    .pip_install(
        "git+https://github.com/huggingface/diffusers.git",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "peft>=0.12.0",
        "huggingface_hub>=0.24.0",
        "safetensors>=0.4.0",
        "sentencepiece",
        "protobuf",
        "Pillow",
        "tqdm",
        "kaggle",
        "pyyaml",
        "datasets>=2.20.0",
        "python-dotenv",
    )
    .add_local_dir("src",          remote_path="/root/project/src")
    .add_local_file("config.yaml", remote_path="/root/project/config.yaml")
    .add_local_file(".env",        remote_path="/root/project/.env")
)

app = modal.App("flux-lora", image=image)


@app.function(
    gpu     = GPU,
    timeout = TIMEOUT,
    volumes = {"/vol": volume},
    secrets = [modal.Secret.from_dict({
        "HF_TOKEN":        HF_TOKEN,
        "KAGGLE_USERNAME": KAGGLE_USERNAME,
        "KAGGLE_KEY":      KAGGLE_KEY,
    })],
)
def train_remote():
    import sys, os
    sys.path.insert(0, "/root/project")

    from src.config import load_config
    from src.dataset import build_dataset
    from src.trainer import FluxLoraTrainer

    cfg = load_config("/root/project/config.yaml")
    cfg.checkpointing.output_dir = OUTPUT_DIR

    dataset = build_dataset(cfg)
    trainer = FluxLoraTrainer(
        cfg        = cfg,
        dataset    = dataset,
        output_dir = OUTPUT_DIR,
        volume     = volume,
    )
    trainer.run()


@app.local_entrypoint()
def main():
    train_remote.remote()
