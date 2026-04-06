import modal
import os
from dotenv import load_dotenv
load_dotenv(".env")

VOLUME_NAME    = "tts-dataset-storage"
GPU            = "H100:1"
TIMEOUT        = 25000
PYTHON_VERSION = "3.11"
OUTPUT_DIR     = "/vol/harmony-tts-output"

HF_TOKEN = os.getenv("HF_TOKEN", "")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

REQUIREMENTS = [
    "torch==2.4.1",
    "torchaudio==2.4.1",
    "accelerate",
    "datasets[audio]",
    "transformers==4.46.1",
    "pydantic==1.10.17",
    "tqdm",
    "soundfile",
    "scipy",
    "pyyaml",
    "protobuf==4.25.8",
    "evaluate",
    "jiwer",
    "librosa",
    "bitsandbytes",
    "huggingface_hub",
    "peft",
    "python-dotenv",
    "wandb",
    "parler-tts @ git+https://github.com/huggingface/parler-tts.git",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python=PYTHON_VERSION,
    )
    .apt_install("git", "ffmpeg", "libsndfile1")
    .run_commands("ulimit -n 65536")
    .pip_install(
        *REQUIREMENTS,
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_dir("src",          remote_path="/root/project/src")
    .add_local_file("config.yaml", remote_path="/root/project/config.yaml")
    .add_local_file(".env",        remote_path="/root/project/.env")
)

app = modal.App("harmony-tts", image=image)


@app.function(
    volumes={"/vol": volume},
    timeout=TIMEOUT,
    gpu=GPU,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN})],
    env={
        "FORCE_LIBSNDFILE":            "1",
        "HF_AUDIO_DISABLE_TORCHCODEC": "1",
    },
)
def train_remote():
    import sys
    sys.path.insert(0, "/root/project")

    from src.config import load_config
    from src.trainer import HarmonyTTSTrainer

    cfg = load_config("/root/project/config.yaml")
    cfg.training.output_dir = OUTPUT_DIR

    trainer = HarmonyTTSTrainer(
        cfg        = cfg,
        output_dir = OUTPUT_DIR,
        volume     = volume,
    )
    trainer.run()


@app.local_entrypoint()
def main():
    train_remote.remote()
