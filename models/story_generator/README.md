# 🧒 Kids Story Generator

An AI pipeline that turns a plain-text educational lesson into a fully structured children's animated story — complete with characters, backgrounds, scene-by-scene passages, and voice scripts — ready to feed into a downstream image/video/TTS production pipeline.

Inference is powered by the **HuggingFace Inference API** (via the OpenAI-compatible router), so **no GPU is required**. Orchestration runs on **Modal** (serverless cloud), which handles scheduling, secrets, and returning outputs to your machine.

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

All LLM calls go through the **HuggingFace Inference Router** using the `openai` Python package. The default model is `Qwen/Qwen2.5-32B-Instruct` served via the `featherless-ai` provider, but this is configurable in `config.yaml`.

---

## Project Structure

```
story_generator/
├── src/
│   ├── generator.py      # Core StoryGenerator class (all HF API calls)
│   ├── config.py         # Config dataclasses + YAML/env loader
│   └── save_outputs.py   # Structures and writes all output files
├── scripts/
│   └── run.py            # Local entrypoint (no GPU required)
├── cloud/
│   └── run.py            # Modal cloud entrypoint
├── API/
│   └── API.py            # Modal-hosted FastAPI endpoint
├── config.yaml           # All runtime settings
├── lesson.txt            # Your input lesson (edit this)
├── requirements.txt
├── Makefile
└── .env.example
```

---

## Requirements

- Python 3.11+
- A [HuggingFace](https://huggingface.co) account with a valid API token (`HF_TOKEN`)
- For cloud execution: a [Modal](https://modal.com) account

No GPU, no local model download, no CUDA required.

---

## Setup

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Set your HuggingFace token:**

```bash
cp .env.example .env
# Edit .env and set: HF_TOKEN=hf_your_token_here
```

**3. Write your lesson:**

Edit `lesson.txt` with the educational content you want turned into a story.

---

## Running

### Local

Runs the full pipeline locally — all LLM calls go out to the HuggingFace API.

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

### Cloud (Modal)

The job runs as a Modal task in the cloud. Modal handles the container, secrets, and result download. All LLM calls still go through the HuggingFace API — no GPU is provisioned.

```bash
# Using Makefile
make modal-run

# Or directly
modal run cloud/run.py --lesson-file lesson.txt
```

### API (Modal-hosted FastAPI)

Deploy a persistent HTTP endpoint that accepts a lesson and returns a zip of all output files.

```bash
modal deploy API/API.py
```

Then POST to the deployed URL:

```bash
curl -X POST https://<your-modal-url>/generate \
  -H "Content-Type: application/json" \
  -d '{"lesson": "Today we learn about the water cycle..."}' \
  --output story.zip
```

---

## Configuration

All settings live in `config.yaml`:

```yaml
credentials:
  hf_token: ""           # Set via HF_TOKEN env var

model:
  id: "Qwen/Qwen2.5-32B-Instruct:featherless-ai"
  hf_base_url: "https://router.huggingface.co/v1"
  max_new_tokens: 4000
  temperature: 0.7
  top_p: 0.9

story:
  min_characters: 2
  max_characters: 6
  min_backgrounds: 2
  max_backgrounds: 6
  min_scenes: 3
  max_scenes: 10

output:
  dir: "outputs"

modal:
  app_name: "kids-story-generator"
  timeout: 3600
  python_version: "3.11"
```

To switch models, change `model.id` to any model available on the HuggingFace router, e.g.:
- `"meta-llama/Llama-3.3-70B-Instruct:fireworks-ai"`
- `"mistralai/Mistral-7B-Instruct-v0.3:hf-inference"`

---

## Output Files

| File | Description |
|---|---|
| `00_generation_manifest.json` | Lists all image prompts (characters + backgrounds), TTS tasks, and frame compositing tasks. |
| `video_timeline.json` | Ordered list of every shot with timing estimates, file paths, and video prompts. |
| `story_index.json` | Complete story structure: all scenes, characters per scene, shots, and lesson elements. |
| `scenes/scene_XX_*/scene.json` | Per-scene data including background, characters with positions, and all shots. |
| `scenes/.../shots/shot_XX_*/prompt.txt` | Video generation prompt for that shot. |
| `scenes/.../shots/shot_XX_*/voice.txt` | TTS input: speaker name, voice description, and dialogue text. |

---

## Notes

- The model performs up to **5 retry attempts** on any JSON-producing prompt before raising an error.
- Each dialogue line in the voice script is constrained to **15–20 words** for TTS pacing.
- Character positions are expressed as `(x, y)` floats from `(0.0, 0.0)` top-left to `(1.0, 1.0)` bottom-right, ready for compositing.
- Characters are always non-human (objects or nature elements) to keep the story grounded in the lesson topic.
