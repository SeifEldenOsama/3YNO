import modal

VOLUME_NAME    = "flux-lora-vol"
GPU            = "H100"
PYTHON_VERSION = "3.11"
TORCH_VERSION  = "2.5.1"
CUDA_VERSION   = "cu124"
LORA_BASE      = "/vol/flux-lora-output"

import os
from dotenv import load_dotenv
load_dotenv(".env")

HF_TOKEN = os.getenv("HF_TOKEN", "")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "libgl1", "libglib2.0-0")
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
        "pyyaml",
        "python-dotenv",
    )
    .add_local_dir("src",          remote_path="/root/project/src")
    .add_local_file("config.yaml", remote_path="/root/project/config.yaml")
    .add_local_file(".env",        remote_path="/root/project/.env")
)

app = modal.App("flux-lora", image=image)


@app.function(
    gpu     = GPU,
    timeout = 60 * 15,
    volumes = {"/vol": volume},
    secrets = [modal.Secret.from_dict({
        "HF_TOKEN": HF_TOKEN,
    })],
)
def inference_remote(
    prompt:              str   = None,
    num_images:          int   = None,
    num_inference_steps: int   = None,
    guidance_scale:      float = None,
    seed:                int   = None,
    lora_path:           str   = None,
):
    import sys, os, glob
    sys.path.insert(0, "/root/project") 

    from src.config import load_config
    from src.inference import FluxLoraInference

    cfg = load_config("/root/project/config.yaml")
    cfg.checkpointing.output_dir = LORA_BASE
    cfg.inference.output_dir     = "/vol/inference_outputs"
    cfg.inference.local_output   = "/vol/inference_outputs"

    checkpoints = sorted(glob.glob(f"{LORA_BASE}/checkpoint-*"))
    resolved    = lora_path or (checkpoints[-1] if checkpoints else LORA_BASE)
    print(f"🔍 Using LoRA path: {resolved}")

    runner = FluxLoraInference(cfg, lora_path=resolved)
    saved  = runner.generate(
        prompt              = prompt,
        num_images          = num_images,
        num_inference_steps = num_inference_steps,
        guidance_scale      = guidance_scale,
        seed                = seed,
        output_dir          = "/vol/inference_outputs",
    )

    volume.commit()
    return saved


@app.local_entrypoint()
def main(
    prompt:              str   = None,
    num_images:          int   = 4,
    num_inference_steps: int   = 28,
    guidance_scale:      float = 3.5,
    seed:                int   = 42,
    lora_path:           str   = None,
):
    saved = inference_remote.remote(
        prompt              = prompt,
        num_images          = num_images,
        num_inference_steps = num_inference_steps,
        guidance_scale      = guidance_scale,
        seed                = seed,
        lora_path           = lora_path,
    )
    print(f"\n🎉 Generated {len(saved)} image(s):")
    for p in saved:
        print(f"   → {p}")
