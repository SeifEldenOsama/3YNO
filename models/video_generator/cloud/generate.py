import modal
import os
import json
import zipfile
import tempfile
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ["HF_TOKEN"]

VOLUME_NAME    = "ltx2-model-cache"
GPU            = "H200"
TIMEOUT        = 7200
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
        "peft", "librosa", "pyyaml", "python-dotenv", "rembg[gpu]",
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

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        prompt:      str = None,
        seed:        int = 42,
    ) -> bytes:
        clip_bytes, _ = self.gen.generate(
            image_bytes = image_bytes,
            audio_bytes = audio_bytes,
            prompt      = prompt,
            seed        = seed,
        )
        return clip_bytes

    @modal.method()
    def generate_scene(self, payload: dict) -> dict:
        import sys
        sys.path.insert(0, "/root/project")
        return self.gen.generate_pipeline(
            shots             = payload["shots"],
            background_images = payload["background_images"],
            character_images  = payload["character_images"],
            audio_files       = payload["audio_files"],
            seed              = payload["seed"],
        )


def _extract_zip(zip_path: str):
    tmp_dir  = Path(tempfile.mkdtemp())
    zip_file = Path(zip_path).resolve()
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(tmp_dir)
    candidates = list(tmp_dir.rglob("shots_flow.json"))
    if not candidates:
        raise FileNotFoundError(f"shots_flow.json not found inside {zip_path}")
    return tmp_dir, candidates[0].parent


def _load_assets(root: Path):
    background_images = {}
    for png in (root / "backgrounds").glob("*.png"):
        background_images[png.stem] = png.read_bytes()

    character_images = {}
    for png in (root / "characters").glob("*.png"):
        character_images[png.stem] = png.read_bytes()

    return background_images, character_images


def _build_scene_payloads(scenes, root, background_images, character_images, seed):
    payloads    = []
    shot_order  = []

    for scene in scenes:
        scene_id    = scene["scene_id"]
        bg_name     = Path(scene["background"]).stem
        scene_chars = scene["characters"]

        scene_shots = []
        scene_audio = {}

        for i, shot in enumerate(scene["shots"]):
            shot_id      = f"s{scene_id}_shot{shot['shot_id']}"
            shot_num     = i + 1
            frame_source = "composite" if shot_num == 1 else "previous_clip"

            scene_shots.append({
                "shot_id":            shot_id,
                "scene_number":       scene_id,
                "shot_number":        shot_num,
                "background_name":    bg_name,
                "frame_source":       frame_source,
                "video_prompt":       shot.get("video_prompt"),
                "speaker":            shot.get("speaker", ""),
                "characters_present": [
                    {"name": c["name"], "position": c["position"]}
                    for c in scene_chars
                ],
            })
            shot_order.append(shot_id)

            voice_path = root / shot["voice_path"]
            if voice_path.exists():
                scene_audio[shot_id] = voice_path.read_bytes()
            else:
                print(f"WARNING: Audio not found for {shot_id}: {voice_path}")

        payloads.append({
            "shots":             scene_shots,
            "background_images": background_images,
            "character_images":  character_images,
            "audio_files":       scene_audio,
            "seed":              seed,
        })

    return payloads, shot_order


def _ensure_ffmpeg():
    import shutil
    if shutil.which("ffmpeg"):
        return
    import sys, urllib.request, zipfile as zf, os
    print("ffmpeg not found — downloading static build...")
    if sys.platform == "win32":
        url      = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        dest_zip = Path(tempfile.gettempdir()) / "ffmpeg.zip"
        urllib.request.urlretrieve(url, dest_zip)
        extract  = Path(tempfile.gettempdir()) / "ffmpeg_bin"
        with zf.ZipFile(dest_zip) as z:
            z.extractall(extract)
        ffmpeg_exe = next(extract.rglob("ffmpeg.exe"))
        os.environ["PATH"] = str(ffmpeg_exe.parent) + os.pathsep + os.environ["PATH"]
        print(f"ffmpeg ready: {ffmpeg_exe}")
    else:
        raise RuntimeError("ffmpeg not found — install it via your package manager")


def _save_and_concat(results, shot_order, clips_dir, tmp_dir, output_path):
    _ensure_ffmpeg()

    clips_path = Path(clips_dir)
    clips_path.mkdir(parents=True, exist_ok=True)

    ordered_clips = []
    for shot_id in shot_order:
        clip_bytes = results.get(shot_id)
        if clip_bytes:
            clip_file = clips_path / f"{shot_id}.mp4"
            clip_file.write_bytes(clip_bytes)
            ordered_clips.append(clip_file)
            print(f"Saved: {clip_file}")
        else:
            print(f"WARNING: No clip for {shot_id}")

    concat_list = Path(tmp_dir) / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in ordered_clips)
    )
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        output_path,
    ], check=True)
    print(f"\nFinal video: {output_path}  ({len(ordered_clips)} clips)")


@app.local_entrypoint()
def main(
    image_path:  str = "test/frame.png",
    audio_path:  str = "test/audio.wav",
    output_path: str = "output.mp4",
    prompt:      str = None,
    seed:        int = 42,
):
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
def run_from_zip(
    zip_path:    str = "outputs.zip",
    output_path: str = "final_video.mp4",
    clips_dir:   str = "outputs/clips",
    seed:        int = 42,
):
    tmp_dir, root = _extract_zip(zip_path)
    shots_flow    = json.loads((root / "shots_flow.json").read_text())
    scenes        = shots_flow["scenes"]
    print(f"Scenes: {len(scenes)}")

    background_images, character_images = _load_assets(root)
    print(f"Backgrounds: {list(background_images)} | Characters: {list(character_images)}")

    payloads, shot_order = _build_scene_payloads(
        scenes, root, background_images, character_images, seed
    )
    print(f"Shots: {len(shot_order)} across {len(payloads)} scenes")

    gen     = VideoGeneratorModal()
    results = {}

    print("\nRunning scenes in parallel...")
    for scene_result in gen.generate_scene.map(payloads):
        results.update(scene_result)

    _save_and_concat(results, shot_order, clips_dir, tmp_dir, output_path)