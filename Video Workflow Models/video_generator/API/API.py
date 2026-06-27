import modal
import io
import os
import json
import zipfile
import tempfile
import subprocess
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from fastapi import HTTPException, UploadFile, File, Form
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
SCALEDOWN      = 60
PYTHON_VERSION = "3.12"
MODEL_CACHE    = "/model-cache"
CLIPS_DIR      = f"{MODEL_CACHE}/clips"   # shared volume path for inter-function clip transfer

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1", "torchaudio==2.5.1", "torchvision==0.20.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "diffusers @ git+https://github.com/huggingface/diffusers.git", "transformers>=4.46.0,<4.52.0", "huggingface_hub[hf_transfer]",
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


@app.function(
    gpu=GPU,
    volumes={MODEL_CACHE: volume},
    timeout=TIMEOUT,
    scaledown_window=SCALEDOWN,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
def generate_scene(payload: dict) -> dict:
    """
    Handles all shots of one scene sequentially.
    Saves each clip to the shared Volume and returns a dict of
    { shot_id -> volume_path } instead of raw bytes, avoiding
    the BlobGet / large-payload transfer issue.
    """
    import sys
    sys.path.insert(0, "/root/project")
    from src.generator import VideoGenerator
    from src.config import load_config

    cfg = load_config("/root/project/config.yaml")
    gen = VideoGenerator(cfg)
    gen.load_model()

    # generate_pipeline returns { shot_id: clip_bytes }
    clip_map: dict = gen.generate_pipeline(
        shots             = payload["shots"],
        background_images = payload["background_images"],
        character_images  = payload["character_images"],
        audio_files       = payload["audio_files"],
        seed              = payload["seed"],
    )

    # --- FIX: write clips to the shared Volume; return paths not bytes ---
    run_id   = payload.get("run_id", "default")
    clips_dir = Path(CLIPS_DIR) / run_id
    clips_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, str] = {}
    for shot_id, clip_bytes in clip_map.items():
        clip_path = clips_dir / f"{shot_id}.mp4"
        clip_path.write_bytes(clip_bytes)
        saved_paths[shot_id] = str(clip_path)
        print(f"  Saved clip to volume: {clip_path} ({len(clip_bytes)/1024:.1f} KB)")

    # Commit so the orchestrator container can immediately read the files
    volume.commit()

    return saved_paths   # small dict of strings — no BlobGet triggered


def _build_scene_payloads(scenes, root, background_images, character_images, seed, run_id):
    payloads   = []
    shot_order = []

    for scene in scenes:
        scene_id      = scene["scene_id"]
        is_host_scene = scene.get("is_host_scene", False)

        # Background: host scenes have no background (None), regular scenes have a path
        bg_raw  = scene.get("background")
        bg_name = Path(bg_raw).stem if bg_raw else None

        scene_chars = scene["characters"]

        scene_shots = []
        scene_audio = {}

        for i, shot in enumerate(scene["shots"]):
            shot_id  = f"s{scene_id}_shot{shot['shot_id']}"
            shot_num = i + 1

            voice_path = root / shot["voice_path"]
            if not voice_path.exists():
                print(f"WARNING: Audio not found for {shot_id}: {voice_path}")
                continue

            scene_shots.append({
                "shot_id":            shot_id,
                "scene_number":       scene_id,
                "shot_number":        shot_num,
                "background_name":    bg_name,
                "video_prompt":       shot.get("video_prompt"),
                "negative_prompt":    shot.get("negative_prompt"),
                "speaker":            shot.get("speaker", ""),
                "is_host_scene":      is_host_scene,
                "characters_present": [
                    {"name": c["name"], "position": c["position"]}
                    for c in scene_chars
                ],
            })
            scene_audio[shot_id] = voice_path.read_bytes()
            shot_order.append(shot_id)

        payloads.append({
            "shots":             scene_shots,
            "background_images": background_images,
            "character_images":  character_images,
            "audio_files":       scene_audio,
            "seed":              seed,
            "run_id":            run_id,   # passed so each run gets its own subfolder
        })

    return payloads, shot_order


@app.function(
    image=image,
    volumes={MODEL_CACHE: volume},   # FIX: mount the volume so clips written by GPU workers are readable here
    timeout=TIMEOUT,
    memory=8192,
)
def run_pipeline_from_zip(zip_bytes: bytes, seed: int = 42) -> bytes:
    import uuid
    tmp_dir = Path(tempfile.mkdtemp())

    # Unique ID for this run to avoid cross-run filename collisions in the volume
    run_id = uuid.uuid4().hex[:12]

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

    payloads, shot_order = _build_scene_payloads(
        scenes, root, background_images, character_images, seed, run_id
    )
    print(f"Total shots: {len(shot_order)} across {len(scenes)} scenes  [run_id={run_id}]")

    # --- FIX: collect volume paths (strings), not raw bytes ---
    path_results: dict[str, str] = {}
    for scene_result in generate_scene.map(payloads):
        path_results.update(scene_result)

    # Reload the volume so freshly-written clips are visible in this container
    volume.reload()

    clips_dir = tmp_dir / "clips"
    clips_dir.mkdir()
    clips = []
    for shot_id in shot_order:
        clip_volume_path = path_results.get(shot_id)
        if not clip_volume_path:
            print(f"WARNING: No clip found for {shot_id}, skipping.")
            continue
        # Read from the shared volume path and write locally for ffmpeg concat
        clip_bytes = Path(clip_volume_path).read_bytes()
        local_path = clips_dir / f"{shot_id}.mp4"
        local_path.write_bytes(clip_bytes)
        clips.append(local_path)

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

    # Clean up this run's clips from the volume to avoid accumulation
    run_clips_dir = Path(CLIPS_DIR) / run_id
    if run_clips_dir.exists():
        import shutil
        shutil.rmtree(run_clips_dir)
        volume.commit()

    return output.read_bytes()


@app.function(
    image=image,
    memory=2048,
    timeout=TIMEOUT,
)
@modal.fastapi_endpoint(method="POST")
async def generate_from_zip(
    story_zip: UploadFile = File(...),
    seed:      int        = Form(default=42),
) -> Response:
    zip_bytes = await story_zip.read()
    if not zip_bytes:
        raise HTTPException(status_code=400, detail="zip file is empty")

    try:
        final_video = run_pipeline_from_zip.remote(zip_bytes, seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(content=final_video, media_type="video/mp4")