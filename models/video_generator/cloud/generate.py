import modal
import os
import json
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
    .add_local_dir("src",          remote_path="/root/project/src")
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

    # ── Single shot (original entrypoint) ────────────────────────────────────
    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        prompt:      str = None,
        seed:        int = 42,
    ) -> bytes:
        """
        Generate one clip from one frame image + one audio file.
        """
        return self.gen.generate(
            image_bytes = image_bytes,
            audio_bytes = audio_bytes,
            prompt      = prompt,
            seed        = seed,
        )

    # ── Full pipeline (scenes + shots) ────────────────────────────────────────
    @modal.method()
    def generate_pipeline(
        self,
        shots:             list,   # video_timeline.json["shots"]
        background_images: dict,   # {background_name: bytes}
        character_images:  dict,   # {character_name: bytes}
        audio_files:       dict,   # {shot_id: bytes}
        seed:              int = 42,
    ) -> dict:
        """
        Process all shots across all scenes.

        Shot 1 of each scene:
          → composite background + characters (white bg removed) at x,y positions
          → feed composited frame to LTX-2

        Shot 2, 3, 4... of same scene:
          → extract last frame of previous clip
          → feed that frame to LTX-2

        Returns {shot_id: clip_bytes} for every shot.
        """
        import sys
        sys.path.insert(0, "/root/project")

        return self.gen.generate_pipeline(
            shots             = shots,
            background_images = background_images,
            character_images  = character_images,
            audio_files       = audio_files,
            seed              = seed,
        )


# ── Single shot entrypoint ────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    image_path:  str = "test/frame.png",
    audio_path:  str = "test/audio.wav",
    output_path: str = "output.mp4",
    prompt:      str = None,
    seed:        int = 42,
):
    """
    Single shot: supply a pre-composited frame.png + audio.wav.
    For full pipeline use generate_pipeline_entrypoint.
    """
    if not Path(image_path).exists() or not Path(audio_path).exists():
        print(f"Files not found: {image_path} or {audio_path}")
        return

    video_bytes = VideoGeneratorModal().generate.remote(
        image_bytes = Path(image_path).read_bytes(),
        audio_bytes = Path(audio_path).read_bytes(),
        prompt      = prompt,
        seed        = seed,
    )
    Path(output_path).write_bytes(video_bytes)
    print(f"Saved: {output_path}")


@app.local_entrypoint()
def run_pipeline(
    timeline_path: str = "outputs/video_timeline.json",
    assets_dir:    str = "outputs",
    audio_dir:     str = "outputs",
    output_dir:    str = "outputs/clips",
    seed:          int = 42,
):
    """
    Full pipeline: reads video_timeline.json and all assets from disk.

    Expected layout (from story generator save_all):
      outputs/
        video_timeline.json
        assets/
          backgrounds/{name}.png
          characters/{name}.png
        scenes/{scene}/shots/{shot}/voice.mp3
    """
    import sys
    sys.path.insert(0, ".")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load timeline ─────────────────────────────────────────────────────────
    with open(timeline_path) as f:
        timeline = json.load(f)
    shots = timeline["shots"]
    print(f"Loaded {len(shots)} shots from {timeline_path}")

    # ── Load background images ────────────────────────────────────────────────
    background_images = {}
    bg_dir = Path(assets_dir) / "assets" / "backgrounds"
    for png in bg_dir.glob("*.png"):
        background_images[png.stem] = png.read_bytes()
        print(f"Background loaded: {png.stem}")

    # ── Load character images ─────────────────────────────────────────────────
    character_images = {}
    char_dir = Path(assets_dir) / "assets" / "characters"
    for png in char_dir.glob("*.png"):
        character_images[png.stem] = png.read_bytes()
        print(f"Character loaded: {png.stem}")

    # ── Load audio files ──────────────────────────────────────────────────────
    audio_files = {}
    for shot in shots:
        shot_id    = shot["shot_id"]
        voice_path = Path(assets_dir) / shot["voice_file"]
        if voice_path.exists():
            audio_files[shot_id] = voice_path.read_bytes()
        else:
            print(f"WARNING: Audio not found for {shot_id}: {voice_path}")

    print(f"\nRunning pipeline: {len(shots)} shots | "
          f"{len(background_images)} backgrounds | "
          f"{len(character_images)} characters | "
          f"{len(audio_files)} audio files")

    # ── Run remote ────────────────────────────────────────────────────────────
    results = VideoGeneratorModal().generate_pipeline.remote(
        shots             = shots,
        background_images = background_images,
        character_images  = character_images,
        audio_files       = audio_files,
        seed              = seed,
    )

    # ── Save clips ────────────────────────────────────────────────────────────
    for shot_id, clip_bytes in results.items():
        out_path = Path(output_dir) / f"{shot_id}.mp4"
        out_path.write_bytes(clip_bytes)
        print(f"Saved: {out_path}")

    print(f"\nDone. {len(results)} clips saved to {output_dir}/")
