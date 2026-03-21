import modal
import os
from dotenv import load_dotenv
load_dotenv(".env")

VOLUME_NAME    = "led-summarizer-vol"
GPU            = "H100"
TIMEOUT        = 86400
PYTHON_VERSION = "3.11"
TORCH_VERSION  = "2.6.0"
CUDA_VERSION   = "cu124"
OUTPUT_DIR     = "/vol/bart-summarizer-output"
CSV_REMOTE     = "/vol/data_summarization.csv"

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
        "datasets>=2.20.0",
        "accelerate>=0.33.0",
        "evaluate>=0.4.0",
        "rouge_score",
        "huggingface_hub>=0.24.0",
        "safetensors>=0.4.0",
        "sentencepiece",
        "pandas",
        "numpy",
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
    timeout = TIMEOUT,
    volumes = {"/vol": volume},
    secrets = [modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN})],
)
def train_remote():
    import sys
    sys.path.insert(0, "/root/project")

    from src.config import load_config
    from src.trainer import BARTSummarizerTrainer

    cfg = load_config("/root/project/config.yaml")
    cfg.output.dir = OUTPUT_DIR

    trainer = BARTSummarizerTrainer(
        cfg        = cfg,
        csv_path   = CSV_REMOTE,
        output_dir = OUTPUT_DIR,
        volume     = volume,
    )
    trainer.run()


@app.local_entrypoint()
def main():
    train_remote.remote()
