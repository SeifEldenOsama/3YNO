import modal
import os
from dotenv import load_dotenv
load_dotenv(".env")

HF_TOKEN     = os.getenv("HF_TOKEN", "")
HF_BASE_URL  = "https://router.huggingface.co/v1"
MODEL_ID     = "Qwen/Qwen2.5-32B-Instruct:featherless-ai"
TIMEOUT      = 3600
PYTHON_VERSION = "3.11"


image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        "openai",
        "pyyaml",
        "python-dotenv",
    )
    .add_local_dir("src",          remote_path="/root/project/src")
    .add_local_file("config.yaml", remote_path="/root/project/config.yaml")
    .add_local_file(".env",        remote_path="/root/project/.env")
)

app = modal.App("kids-story-generator", image=image)


@app.cls(
    timeout=TIMEOUT,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN})],
)
class StoryGeneratorModal:

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

    @modal.method()
    def run(self, lesson: str) -> dict:
        import sys
        sys.path.insert(0, "/root/project")

        gen = self.gen

        print("Stage 0/4 — Analyzing lesson...", flush=True)
        scale           = gen.analyze_lesson(lesson)
        num_characters  = scale["num_characters"]
        num_backgrounds = scale["num_backgrounds"]
        num_scenes      = scale["num_scenes"]
        lesson_steps    = scale["lesson_steps"]
        print(f"{num_characters} characters, {num_backgrounds} backgrounds, {num_scenes} scenes", flush=True)

        print("Stage 1a/4 — Generating characters...", flush=True)
        characters = gen.generate_characters(lesson, num_characters, lesson_steps)
        print(f"{len(characters)} characters generated.", flush=True)

        print("Stage 1b/4 — Generating backgrounds...", flush=True)
        backgrounds = gen.generate_backgrounds(lesson, num_backgrounds, lesson_steps)
        print(f"{len(backgrounds)} backgrounds generated.", flush=True)

        print("Stage 2/4 — Generating story outline...", flush=True)
        outline = gen.generate_outline(lesson, characters, backgrounds, num_scenes, lesson_steps)
        print(f"{len(outline)} scenes planned.", flush=True)

        print("Stage 3/4 — Writing story passages...", flush=True)
        passages = gen.generate_passages(lesson, outline, characters, backgrounds, lesson_steps)
        print(f"{len(passages)} passages written.", flush=True)

        print("Stage 4/4 — Generating voice scripts...", flush=True)
        scripts = gen.generate_voice_scripts(passages, characters)
        print("Voice scripts done.", flush=True)

        return {
            "lesson":         lesson,
            "lesson_steps":   lesson_steps,
            "characters":     characters,
            "backgrounds":    backgrounds,
            "outline":        outline,
            "story_passages": passages,
            "voice_scripts":  scripts,
        }


@app.local_entrypoint()
def main(lesson_file: str = "lesson.txt"):
    import sys
    sys.path.insert(0, ".")

    from src.save_outputs import save_all

    if not os.path.exists(lesson_file):
        print(f"Lesson file not found: {lesson_file}")
        print("Create a lesson.txt file with your lesson text.")
        return

    with open(lesson_file, "r", encoding="utf-8") as f:
        lesson = f.read().strip()

    print(f"Lesson loaded from: {lesson_file}")
    print(f"Lesson length: {len(lesson.split())} words")

    generator = StoryGeneratorModal()
    result    = generator.run.remote(lesson)

    print("Saving output files...")
    save_all(result, out_dir="outputs")
    print("Done! Check the outputs/ folder.")
