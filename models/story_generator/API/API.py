import modal
import os
import shutil
import subprocess
import tempfile
from pydantic import BaseModel
from fastapi.responses import FileResponse
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HF_BASE_URL  = "https://router.huggingface.co/v1"
MODEL_ID     = "Qwen/Qwen2.5-32B-Instruct:featherless-ai"
TIMEOUT      = 3600
PYTHON_VERSION = "3.11"

# Register the HF token as a Modal secret on first deploy
if not os.environ.get("MODAL_TASK_ID") and HF_TOKEN:
    subprocess.run(
        ["modal", "secret", "create", "my-huggingface-secret", f"HF_TOKEN={HF_TOKEN}", "--force"],
        check=True,
    )

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        "openai",
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
    timeout=TIMEOUT,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
class StoryGeneratorAPI:

    @modal.enter()
    def setup(self):
        import sys
        sys.path.insert(0, "/root/project")

        from src.generator import StoryGenerator
        self.gen = StoryGenerator()
        self.gen.load_model(
            hf_token    = os.environ["HF_TOKEN"],
            model_id    = MODEL_ID,
            hf_base_url = HF_BASE_URL,
        )
        print("HuggingFace API client ready!", flush=True)

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
        out_dir  = os.path.join(temp_dir, "output")
        os.makedirs(out_dir, exist_ok=True)

        save_all(result, out_dir=out_dir)

        zip_filename = "output"
        zip_path   = os.path.join(temp_dir, zip_filename)
        shutil.make_archive(zip_path, "zip", out_dir)

        print("Done!", flush=True)

        return FileResponse(
            path       = zip_path + ".zip",
            filename   = "output.zip",
            media_type = "application/zip",
        )