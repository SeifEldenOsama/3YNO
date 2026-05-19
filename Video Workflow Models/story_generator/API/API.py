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

        print("Stage 5/5 — Organizing files and zipping...", flush=True)

        from src.save_outputs import save_all

        temp_dir   = tempfile.mkdtemp()
        story_name = lesson.replace(" ", "_")[:30]
        out_dir    = os.path.join(temp_dir, story_name)
        os.makedirs(out_dir)

        save_all(
            result  = {
                "characters":   characters,
                "backgrounds":  backgrounds,
                "voice_scripts": scripts,
            },
            out_dir = out_dir,
        )

        zip_path = os.path.join(temp_dir, f"{story_name}.zip")
        shutil.make_archive(os.path.join(temp_dir, story_name), "zip", temp_dir, story_name)

        print("Done!", flush=True)

        return FileResponse(
            path       = zip_path,
            filename   = f"{story_name}.zip",
            media_type = "application/zip",
        )