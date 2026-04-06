import modal
import io
import os
import json
import zipfile
import tempfile
import subprocess
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import Response

load_dotenv(find_dotenv())

if not os.environ.get("MODAL_TASK_ID"):
    HF_TOKEN = os.environ["HF_TOKEN"]
    subprocess.run(
        ["modal", "secret", "create", "my-huggingface-secret", f"HF_TOKEN={HF_TOKEN}", "--force"],
        check=True
    )

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
        "fastapi[standard]",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF":   "expandable_segments:True",
    })
    .add_local_dir("src",          remote_path="/root/project/src")
    .add_local_file("config.yaml", remote_path="/root/project/config.yaml")
)

app = modal.App("video-generator-api", image=image)



@app.cls(
    gpu=GPU,
    volumes={MODEL_CACHE: volume},
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
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
        print("Model ready ✓")

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        prompt:      str = None,
        seed:        int = 42,
    ) -> bytes:
        return self.gen.generate(
            image_bytes=image_bytes,
            audio_bytes=audio_bytes,
            prompt=prompt,
            seed=seed,
        )

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



@app.function(
    image=image,
    timeout=TIMEOUT,
    memory=8192,
)
def run_pipeline_from_zip(zip_bytes: bytes, seed: int = 42) -> bytes:
    tmp_dir = Path(tempfile.mkdtemp())

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(tmp_dir)

    candidates = list(tmp_dir.rglob("shots_flow.json"))
    if not candidates:
        raise FileNotFoundError("shots_flow.json not found inside zip")
    root = candidates[0].parent

    background_images = {
        png.stem: png.read_bytes()
        for png in (root / "backgrounds").glob("*.png")
    }
    character_images = {
        png.stem: png.read_bytes()
        for png in (root / "characters").glob("*.png")
    }

    shots_flow = json.loads((root / "shots_flow.json").read_text())
    scenes     = shots_flow["scenes"]

    # Build payloads
    payloads   = []
    shot_order = []
    for scene in scenes:
        scene_id    = scene["scene_id"]
        bg_name     = Path(scene["background"]).stem
        scene_chars = scene["characters"]
        scene_shots = []
        scene_audio = {}
        for i, shot in enumerate(scene["shots"]):
            shot_id      = f"s{scene_id}_shot{shot['shot_id']}"
            frame_source = "composite" if i == 0 else "previous_clip"
            scene_shots.append({
                "shot_id":            shot_id,
                "scene_number":       scene_id,
                "shot_number":        i + 1,
                "background_name":    bg_name,
                "frame_source":       frame_source,
                "video_prompt":       shot.get("video_prompt"),
                "negative_prompt":    shot.get("negative_prompt"),
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

        payloads.append({
            "shots":             scene_shots,
            "background_images": background_images,
            "character_images":  character_images,
            "audio_files":       scene_audio,
            "seed":              seed,
        })

    gen     = VideoGeneratorModal()
    results = {}
    for scene_result in gen.generate_scene.map(payloads):
        results.update(scene_result)

    clips_dir = tmp_dir / "clips"
    clips_dir.mkdir()
    clips = []
    for shot_id in shot_order:
        clip_bytes = results.get(shot_id)
        if clip_bytes:
            p = clips_dir / f"{shot_id}.mp4"
            p.write_bytes(clip_bytes)
            clips.append(p)

    if not clips:
        raise ValueError("No clips were generated")

    concat_list = tmp_dir / "concat.txt"
    concat_list.write_text("\n".join(f"file '{p.resolve()}'" for p in clips))

    output = tmp_dir / "final.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(output),
    ], check=True)

    return output.read_bytes()



web_app = FastAPI(title="Video Generator API")


@app.function(
    image=image,
    memory=2048, 
    timeout=TIMEOUT,
)
@modal.asgi_app()
def fastapi_app():

    @web_app.post(
        "/generate",
        response_class=Response,
        summary="Generate a single clip from an image + audio",
    )
    async def generate(
        image: UploadFile = File(...),
        audio: UploadFile = File(...),
        prompt: str       = Form(default=None),
        seed:   int       = Form(default=42),
    ):
        image_bytes = await image.read()
        audio_bytes = await audio.read()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="image file is empty")
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="audio file is empty")

        video_bytes = VideoGeneratorModal().generate.remote(
            image_bytes=image_bytes,
            audio_bytes=audio_bytes,
            prompt=prompt,
            seed=seed,
        )
        return Response(content=video_bytes, media_type="video/mp4")

    @web_app.post(
        "/generate-from-zip",
        response_class=Response,
        summary="Generate full video from story zip",
    )
    async def generate_from_zip(
        story_zip: UploadFile = File(...),
        seed:      int        = Form(default=42),
    ):
        zip_bytes = await story_zip.read()
        if not zip_bytes:
            raise HTTPException(status_code=400, detail="zip file is empty")

        try:
            final_video = run_pipeline_from_zip.remote(zip_bytes, seed)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return Response(content=final_video, media_type="video/mp4")

    @web_app.get("/health")
    async def health():
        return {"status": "ok"}

    return web_app