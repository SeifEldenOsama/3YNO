import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.generator import StoryGenerator
from src.save_outputs import save_all


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--lesson", default="lesson.txt", help="Path to lesson text file")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if not os.path.exists(args.lesson):
        print(f"Lesson file not found: {args.lesson}")
        return

    with open(args.lesson, "r", encoding="utf-8") as f:
        lesson = f.read().strip()

    out_dir = args.output or cfg.output.dir

    gen = StoryGenerator()
    gen.load_model(
        hf_token    = cfg.credentials.hf_token,
        model_id    = cfg.model.id,
        hf_base_url = cfg.model.hf_base_url,
    )

    scale           = gen.analyze_lesson(lesson)
    lesson_steps    = scale["lesson_steps"]
    num_characters  = scale["num_characters"]
    num_backgrounds = scale["num_backgrounds"]
    num_scenes      = scale["num_scenes"]

    characters  = gen.generate_characters(lesson, num_characters, lesson_steps)
    backgrounds = gen.generate_backgrounds(lesson, num_backgrounds, lesson_steps)
    outline     = gen.generate_outline(lesson, characters, backgrounds, num_scenes, lesson_steps)
    passages    = gen.generate_passages(lesson, outline, characters, backgrounds, lesson_steps)
    scripts     = gen.generate_voice_scripts(passages, characters)

    result = {
        "lesson":         lesson,
        "lesson_steps":   lesson_steps,
        "characters":     characters,
        "backgrounds":    backgrounds,
        "outline":        outline,
        "story_passages": passages,
        "voice_scripts":  scripts,
    }

    save_all(result, out_dir=out_dir)


if __name__ == "__main__":
    main()
