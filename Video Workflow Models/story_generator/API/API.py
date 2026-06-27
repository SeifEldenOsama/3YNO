import modal
import os
import json
import shutil
import tempfile
from pydantic import BaseModel
from fastapi.responses import FileResponse
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

VOLUME_NAME = "story-model-cache"
GPU         = "H100"
TIMEOUT     = 3600
MODEL_ID    = "Qwen/Qwen2.5-32B-Instruct"
CACHE_DIR   = "/model-cache"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "bitsandbytes==0.43.3",
        "huggingface_hub==0.24.6",
        "scipy",
        "pyyaml",
        "python-dotenv",
        "fastapi[standard]",
        "pydantic",
        "openai",
    )
    .add_local_dir("src",          remote_path="/root/project/src")
    .add_local_file("config.yaml", remote_path="/root/project/config.yaml")
    .add_local_file(".env",        remote_path="/root/project/.env")
    .add_local_file("3YNO.png",    remote_path="/root/project/3YNO.png")
)

app = modal.App("kids-story-generator-api", image=image)


class Query(BaseModel):
    lesson: str


@app.cls(
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={CACHE_DIR: volume},
    scaledown_window=30,
)
class StoryGeneratorAPI:

    @modal.enter()
    def load_model(self):
        import sys
        sys.path.insert(0, "/root/project")

        from dotenv import load_dotenv
        load_dotenv("/root/project/.env")

        from src.generator import StoryGenerator
        self.gen = StoryGenerator()
        self.gen.load_model(
            model_id  = MODEL_ID,
            cache_dir = CACHE_DIR,
            hf_token  = os.environ["HF_TOKEN"],
        )
        print("Model loaded and ready!", flush=True)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, query: Query):
        import sys
        sys.path.insert(0, "/root/project")

        lesson = query.lesson.strip()
        if not lesson:
            return {"error": "lesson text is empty"}

        gen = self.gen

        print("Stage 0/4 — Analyzing lesson...", flush=True)
        scale           = gen.analyze_lesson(lesson)
        num_characters  = scale["num_characters"]
        num_backgrounds = scale["num_backgrounds"]
        num_scenes      = scale["num_scenes"]
        lesson_steps    = scale["lesson_steps"]

        print("Stage 1a/4 — Generating characters...", flush=True)
        characters = gen.generate_characters(lesson, num_characters, lesson_steps)

        print("Stage 1b/4 — Generating backgrounds...", flush=True)
        backgrounds = gen.generate_backgrounds(lesson, num_backgrounds, lesson_steps)

        print("Stage 2/4 — Generating outline...", flush=True)
        outline = gen.generate_outline(lesson, characters, backgrounds, num_scenes, lesson_steps)

        print("Stage 3/4 — Writing passages...", flush=True)
        passages = gen.generate_passages(lesson, outline, characters, backgrounds, lesson_steps)

        print("Stage 4/4 — Generating voice scripts...", flush=True)
        scripts = gen.generate_voice_scripts(passages, characters)

        print("Stage 4.5/5 — Generating 3YNO host scenes...", flush=True)
        host_scenes = gen.generate_3yno_scenes(lesson, scripts)

        print("Stage 5/5 — Organizing files and zipping...", flush=True)

        from src.save_outputs import save_all
        import shutil

        # Create a temporary container directory
        temp_dir = tempfile.mkdtemp()
        
        # Enforce target directory to be named exactly "outputs"
        out_dir = os.path.join(temp_dir, "outputs")
        os.makedirs(out_dir)

        # Save generated objects straight into the "outputs" directory
        save_all(
            result  = {
                "characters":    characters,
                "backgrounds":   backgrounds,
                "voice_scripts": scripts,
                "host_scenes":   host_scenes,
            },
            out_dir = out_dir,
        )

        # Auto-bundle the fixed 3YNO image — no manual step needed.
        # The PNG is embedded into the Modal image at deploy time via
        # .add_local_file("3YNO.png", ...) so it is always available.
        zyno_src = "/root/project/3YNO.png"
        zyno_dst = os.path.join(out_dir, "characters", "3YNO.png")
        if os.path.exists(zyno_src):
            shutil.copy(zyno_src, zyno_dst)
            print("3YNO.png bundled into output automatically.", flush=True)
        else:
            print(
                "WARNING: 3YNO.png not found at /root/project/3YNO.png. "
                "Place 3YNO.png in story_generator/ and redeploy.",
                flush=True,
            )

        zip_path = os.path.join(temp_dir, "outputs.zip")
        
        # Zips files located directly inside out_dir, omitting any parent directory paths
        shutil.make_archive(os.path.join(temp_dir, "outputs"), "zip", out_dir)

        print("Done!", flush=True)

        return FileResponse(
            path       = zip_path,
            filename   = "outputs.zip",
            media_type = "application/zip",
        )