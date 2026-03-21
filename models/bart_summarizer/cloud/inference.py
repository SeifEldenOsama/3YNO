import modal
import os
from dotenv import load_dotenv
load_dotenv(".env")

VOLUME_NAME    = "led-summarizer-vol"
GPU            = "H100"
PYTHON_VERSION = "3.11"
TORCH_VERSION  = "2.6.0"
CUDA_VERSION   = "cu124"
MODEL_DIR      = "/vol/bart-summarizer-output"

HF_TOKEN = os.getenv("HF_TOKEN", "")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        "torchvision",
        index_url=f"https://download.pytorch.org/whl/{CUDA_VERSION}",
    )
    .pip_install(
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "safetensors>=0.4.0",
        "sentencepiece",
        "pandas",
        "pyyaml",
        "python-dotenv",
    )
    .add_local_dir("src",          remote_path="/root/project/src")
    .add_local_file("config.yaml", remote_path="/root/project/config.yaml")
    .add_local_file(".env",        remote_path="/root/project/.env")
)

app = modal.App("bart-summarizer", image=image)


@app.function(
    gpu     = GPU,
    timeout = 60 * 10,
    volumes = {"/vol": volume},
    secrets = [modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN})],
)
def summarize_remote(text: str) -> str:
    import sys
    sys.path.insert(0, "/root/project")

    from src.config import load_config
    from src.inference import BARTSummarizerInference

    cfg    = load_config("/root/project/config.yaml")
    runner = BARTSummarizerInference(cfg, model_path=MODEL_DIR)
    return runner.summarize(text)


@app.local_entrypoint()
def main(text: str = ""):
    summary = summarize_remote.remote(text)
    print(summary)
