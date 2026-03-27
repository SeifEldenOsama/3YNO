import modal
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]

VOLUME_NAME    = "ltx2-model-cache"
GPU            = "H200"
TIMEOUT        = 3600
SCALEDOWN      = 3600
PYTHON_VERSION = "3.12"
MODEL_CACHE    = "/model-cache"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1", "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "diffusers", "transformers==4.52.4", "huggingface_hub[hf_transfer]",
        "hf_transfer", "sentencepiece", "numpy", "pillow", "soundfile",
        "imageio[ffmpeg]", "accelerate", "einops", "scipy", "av", "moviepy",
        "peft", "librosa", "pyyaml", "python-dotenv",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF":   "expandable_segments:True",
    })
    .add_local_dir("src", remote_path="/root/project/src")
    .add_local_file("config.yaml", remote_path="/root/project/config.yaml")
)

app = modal.App("video-generator", image=image)


@app.cls(
    gpu=GPU,
    volumes={MODEL_CACHE: volume},
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN})],
)
class VideoGeneratorModal:

    @modal.enter()
    def load_model(self):
        import sys
        sys.path.insert(0, "/root/project")

        from src.generator import VideoGenerator
        from src.config import load_config

        cfg      = load_config("/root/project/config.yaml")
        self.gen = VideoGenerator(cfg)
        self.gen.load_model()

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        prompt:      str  = None,
        seed:        int  = 42,
    ) -> bytes:
        return self.gen.generate(
            image_bytes=image_bytes,
            audio_bytes=audio_bytes,
            prompt=prompt,
            seed=seed,
        )


@app.local_entrypoint()
def main(
    image_path:  str = "test/droplet.png",
    audio_path:  str = "test/surprised1.wav",
    output_path: str = "output_hq.mp4",
    prompt: str = (
        "A cute animated water droplet character with a round expressive face, "
        "big glossy eyes blinking naturally, mouth and lips moving clearly in sync with speech, "
        "subtle bouncing and swaying motion, surrounded by a vibrant underwater ocean scene "
        "with colorful tropical fish swimming around, soft blue-green water caustics lighting, "
        "coral reef in background, bubbles rising, smooth fluid animation, high quality"
    ),
    seed: int = 42,
):
    if not Path(image_path).exists() or not Path(audio_path).exists():
        print(f"Files not found: {image_path} or {audio_path}")
        return

    image_bytes = Path(image_path).read_bytes()
    audio_bytes = Path(audio_path).read_bytes()

    video_bytes = VideoGeneratorModal().generate.remote(
        image_bytes=image_bytes,
        audio_bytes=audio_bytes,
        prompt=prompt,
        seed=seed,
    )

    Path(output_path).write_bytes(video_bytes)
    print(f"✨ Saved to {output_path}")
