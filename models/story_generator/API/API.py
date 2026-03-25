import modal
import os
import shutil
import tempfile
from pydantic import BaseModel
from fastapi.responses import FileResponse

VOLUME_NAME    = "story-model-cache"
GPU            = "A100"
TIMEOUT        = 3600
PYTHON_VERSION = "3.11"
MODEL_ID       = "Qwen/Qwen2.5-32B-Instruct"
CACHE_DIR      = "/model-cache"

HF_TOKEN = "YOUR TOKEN HERE"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
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
    scaledown_window=300,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN})],
)
class StoryGeneratorAPI:

    @modal.enter()
    def load_model(self):
        import sys
        sys.path.insert(0, "/root/project")

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

        from src.save_outputs import save_all

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
        outline = gen.generate_outline(
            lesson, characters, backgrounds, num_scenes, lesson_steps
        )

        print("Stage 3/4 — Writing passages...", flush=True)
        passages = gen.generate_passages(
            lesson, outline, characters, backgrounds, lesson_steps
        )

        print("Stage 4/4 — Generating voice scripts...", flush=True)
        scripts = gen.generate_voice_scripts(passages, characters)

        result = {
            "lesson":         lesson,
            "lesson_steps":   lesson_steps,
            "characters":     characters,
            "backgrounds":    backgrounds,
            "outline":        outline,
            "story_passages": passages,
            "voice_scripts":  scripts,
        }

        print("Saving output files...", flush=True)
        temp_dir = tempfile.mkdtemp()
        out_dir  = os.path.join(temp_dir, "story_output")
        os.makedirs(out_dir, exist_ok=True)

        save_all(result, out_dir=out_dir)

        story_name = lesson.strip().replace(" ", "_")[:30]
        zip_path   = os.path.join(temp_dir, story_name)
        shutil.make_archive(zip_path, "zip", out_dir)

        print("Done!", flush=True)

        return FileResponse(
            path       = zip_path + ".zip",
            filename   = f"{story_name}.zip",
            media_type = "application/zip",
        )