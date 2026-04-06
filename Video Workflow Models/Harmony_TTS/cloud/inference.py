import modal
import os
from dotenv import load_dotenv
load_dotenv(".env")

VOLUME_NAME    = "tts-dataset-storage"
GPU            = "H100:1"
PYTHON_VERSION = "3.11"
MODEL_DIR      = "/vol/harmony-tts-output"

HF_TOKEN = os.getenv("HF_TOKEN", "")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python=PYTHON_VERSION,
    )
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        "transformers==4.46.1",
        "soundfile",
        "scipy",
        "pyyaml",
        "python-dotenv",
        "parler-tts @ git+https://github.com/huggingface/parler-tts.git",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_dir("src",          remote_path="/root/project/src")
    .add_local_file("config.yaml", remote_path="/root/project/config.yaml")
    .add_local_file(".env",        remote_path="/root/project/.env")
)

app = modal.App("harmony-tts", image=image)


@app.function(
    volumes={"/vol": volume},
    timeout=60 * 10,
    gpu=GPU,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN})],
    env={
        "FORCE_LIBSNDFILE":            "1",
        "HF_AUDIO_DISABLE_TORCHCODEC": "1",
    },
)
def generate_remote(
    text:        str = "Hello, this is Harmony TTS speaking.",
    description: str = "A female speaker delivers a cheerful and clear speech.",
) -> bytes:
    import sys
    sys.path.insert(0, "/root/project")

    from src.config import load_config
    from src.inference import HarmonyTTSInference

    cfg    = load_config("/root/project/config.yaml")
    runner = HarmonyTTSInference(cfg, model_path=MODEL_DIR)
    path   = runner.generate(text, description, output_path="/tmp/output.wav")

    with open(path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    text:        str = "Hello, this is Harmony TTS speaking.",
    description: str = "A female speaker delivers a cheerful and clear speech.",
    output:      str = "output.wav",
):
    audio_bytes = generate_remote.remote(text=text, description=description)
    with open(output, "wb") as f:
        f.write(audio_bytes)
    print(f"Audio saved to: {output}")
