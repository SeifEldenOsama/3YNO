# 🧒 Kids Story Generator

An AI pipeline that turns a plain-text educational lesson into a fully structured children's animated story — complete with characters, backgrounds, scene-by-scene passages, and voice scripts — ready to feed into a downstream image/video/TTS production pipeline.

---

## How It Works

The pipeline processes a lesson in five sequential stages:

```
lesson.txt
    │
    ▼
1. Analyze Lesson       → decides num of characters, backgrounds & scenes
    │
    ▼
2. Generate Characters  → cartoon animal/object characters tied to lesson concepts
    │
    ▼
3. Generate Backgrounds → scene settings matching each learning step
    │
    ▼
4. Generate Outline     → scene-by-scene story plan covering every lesson step
    │
    ▼
5. Write Passages       → narrative text (age 5–8, simple language)
    │
    ▼
6. Generate Voice Scripts → shot-by-shot dialogue with speaker, voice tone & video prompts
    │
    ▼
outputs/
  ├── 00_generation_manifest.json   ← asset generation tasks (images, voices, frames)
  ├── video_timeline.json           ← ordered shot list for video assembly
  ├── story_index.json              ← full story with all scenes & shots
  └── scenes/
        └── scene_XX_<title>/
              ├── scene.json
              └── shots/
                    └── shot_XX_<speaker>/
                          ├── prompt.txt   ← video generation prompt
                          └── voice.txt    ← TTS input
```

The model used is **Qwen/Qwen2.5-32B-Instruct**, loaded in 4-bit quantization (NF4) via `bitsandbytes`.

---

## Project Structure

```
story_generator/
├── src/
│   ├── generator.py      # Core StoryGenerator class (all LLM calls)
│   ├── config.py         # Config dataclasses + YAML/env loader
│   └── save_outputs.py   # Structures and writes all output files
├── scripts/
│   ├── run.py            # Local entrypoint 
├── cloud/
│   └── run.py            # Modal (cloud GPU) entrypoint
├── config.yaml           # All runtime settings
├── lesson.txt            # Your input lesson (edit this)
├── requirements.txt
├── Makefile
└── .env.example
```

---

## Requirements

- Python 3.11+
- CUDA-capable GPU with at least ~40 GB VRAM (A100 recommended)
- A [Hugging Face](https://huggingface.co) account with access to the model
- For cloud execution: a [Modal](https://modal.com) account

---

## Setup

**1. Clone and install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Set your Hugging Face token:**

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_your_token_here
```

**3. Write your lesson:**

Edit `lesson.txt` with the educational content you want turned into a story. The included example teaches the water cycle to children aged 5–8.

---

## Running

### Local (requires a GPU machine)

```bash
# Using Makefile
make run

# Or directly
python scripts/run.py --lesson lesson.txt
```

Optional flags:
```
--config   Path to config.yaml (default: config.yaml)
--lesson   Path to lesson text file (default: lesson.txt)
--output   Output directory (default: value from config.yaml)
```

### Cloud (Modal — A100 GPU)

```bash
# Using Makefile
make modal-run

# Or directly
modal run cloud/run.py --lesson-file lesson.txt
```

The cloud runner spins up an A100 on Modal, loads the model into a persistent volume (`story-model-cache`) so it is only downloaded once, runs the full pipeline, and saves all outputs locally to `outputs/`.


## Configuration

All settings live in `config.yaml`:

```yaml
credentials:
  hf_token: ""           # Override via HF_TOKEN env var

model:
  id: "Qwen/Qwen2.5-32B-Instruct"
  cache_dir: "/model-cache"
  max_new_tokens: 4000
  temperature: 0.7
  top_p: 0.9

story:
  min_characters: 2      # Pipeline auto-selects within these bounds
  max_characters: 6
  min_backgrounds: 2
  max_backgrounds: 6
  min_scenes: 3
  max_scenes: 10

output:
  dir: "outputs"

modal:
  app_name: "kids-story-generator"
  volume_name: "story-model-cache"
  gpu: "A100"
  timeout: 3600
  python_version: "3.11"
```

`HF_TOKEN` can be set in `.env` or as an environment variable — it takes priority over `config.yaml`.

---

## Output Files

| File | Description |
|---|---|
| `00_generation_manifest.json` | Lists all image prompts (characters + backgrounds), TTS tasks, and frame compositing tasks. Feed this into your image/voice generation tools. |
| `video_timeline.json` | Ordered list of every shot with timing estimates, file paths, and video prompts. Use this to assemble the final video. |
| `story_index.json` | Complete story structure: all scenes, characters per scene, shots, and lesson elements. |
| `scenes/scene_XX_*/scene.json` | Per-scene data including background, characters with positions, and all shots. |
| `scenes/.../shots/shot_XX_*/prompt.txt` | Video generation prompt for that shot. |
| `scenes/.../shots/shot_XX_*/voice.txt` | TTS input: speaker name, voice description, and dialogue text. |

Character images go to `assets/characters/<name>.png` and backgrounds to `assets/backgrounds/<name>.png` — these paths are referenced throughout the manifests but must be generated by a separate image generation step.

---

## Example Lesson

The included `lesson.txt` teaches the water cycle (evaporation → condensation → precipitation) to children aged 5–8. The pipeline will automatically create characters like a sun and a raindrop, backgrounds like an ocean shore and a cloudy sky, and a multi-scene story where each character explains their role in the cycle through dialogue.

---

## Notes

- The model performs up to **5 retry attempts** on any JSON-producing prompt before raising an error.
- Each dialogue line in the voice script is constrained to **15–20 words** for TTS pacing.
- Character positions are expressed as `(x, y)` floats from `(0.0, 0.0)` top-left to `(1.0, 1.0)` bottom-right, ready for compositing.
- Characters are always non-human (objects or nature elements) to keep the story grounded in the lesson topic.
