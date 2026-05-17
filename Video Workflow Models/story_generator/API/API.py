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
    scaledown_window=300,
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

        temp_dir   = tempfile.mkdtemp()
        story_name = lesson.replace(" ", "_")[:30]
        base_path  = os.path.join(temp_dir, story_name)
        os.makedirs(base_path)

        full_data = {
            "lesson":         lesson,
            "lesson_steps":   lesson_steps,
            "characters":     characters,
            "backgrounds":    backgrounds,
            "outline":        outline,
            "story_passages": passages,
            "voice_scripts":  scripts,
        }

        with open(os.path.join(base_path, "story_index.json"), "w") as f:
            json.dump(full_data, f, indent=2)

        assets_dir = os.path.join(base_path, "assets")
        os.makedirs(os.path.join(assets_dir, "characters"),  exist_ok=True)
        os.makedirs(os.path.join(assets_dir, "backgrounds"), exist_ok=True)

        with open(os.path.join(assets_dir, "characters", "characters.json"), "w") as f:
            json.dump(characters, f, indent=2)
        with open(os.path.join(assets_dir, "backgrounds", "backgrounds.json"), "w") as f:
            json.dump(backgrounds, f, indent=2)

        scenes_dir       = os.path.join(base_path, "scenes")
        scripts_by_scene = {s["scene_number"]: s["script"] for s in scripts}

        for scene in outline:
            scene_num   = scene["scene_number"]
            scene_title = scene["title"].replace(" ", "_").replace("'", "")
            scene_path  = os.path.join(scenes_dir, f"scene_{scene_num:02d}_{scene_title}")
            shots_path  = os.path.join(scene_path, "shots")
            os.makedirs(shots_path, exist_ok=True)

            with open(os.path.join(scene_path, "scene.json"), "w") as f:
                json.dump(scene, f, indent=2)

            scene_script = scripts_by_scene.get(scene_num, [])
            for i, shot in enumerate(scene_script, 1):
                shot_name = f"shot_{i:02d}_{shot['speaker']}"
                shot_path = os.path.join(shots_path, shot_name)
                os.makedirs(shot_path, exist_ok=True)

                with open(os.path.join(shot_path, "voice.txt"), "w") as f:
                    f.write(shot["text"])
                with open(os.path.join(shot_path, "prompt.txt"), "w") as f:
                    f.write(shot["video_prompt"])
                with open(os.path.join(shot_path, "metadata.json"), "w") as f:
                    json.dump(shot, f, indent=2)

        zip_path = os.path.join(temp_dir, f"{story_name}.zip")
        shutil.make_archive(os.path.join(temp_dir, story_name), "zip", base_path)

        print("Done!", flush=True)

        return FileResponse(
            path       = zip_path,
            filename   = f"{story_name}.zip",
            media_type = "application/zip",
        )