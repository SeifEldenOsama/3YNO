# 🧒 Kids Story Generator

An AI pipeline that turns a plain-text educational lesson into a fully structured children's animated story — complete with characters, backgrounds, scene-by-scene passages, and voice scripts — ready to feed into a downstream image/video/TTS production pipeline.

Text generation uses **Qwen/Qwen2.5-32B-Instruct**. `config.yaml` is set up to call it through the **HuggingFace Inference Router** (OpenAI-compatible), but the current `StoryGenerator.load_model()` implementation actually downloads and runs the model **locally** via `transformers`, and both the Modal cloud run (`cloud/run.py`) and the deployed API (`API/API.py`) explicitly request an **H100 GPU** — a GPU is required end-to-end today, despite the router-style config. Orchestration runs on **Modal** (serverless cloud), which handles scheduling, secrets, and returning outputs to your machine.

---

## How It Works

The pipeline processes a lesson in six sequential stages:

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
output/
  ├── characters.json     ← character list with name, visual description, output image path
  ├── backgrounds.json    ← background list with name, visual description, output image path
  ├── voices.json         ← every voice line (regular scenes + 3YNO host lines) with text, voice description, output audio path
  ├── shots_flow.json     ← full ordered scene/shot manifest (background, characters, positions, video prompt, voice path per shot)
  ├── characters/         ← empty folder, ready for generated character PNGs
  ├── backgrounds/        ← empty folder, ready for generated background PNGs
  └── voices/             ← empty folder, ready for generated voice WAVs
```

The API deployment (`API/API.py`) additionally runs a `generate_3yno_scenes` step that inserts intro/transition/outro narration lines for the fixed 3YNO host character; the local script (`scripts/run.py`) does not currently call this step, so its output omits the host scenes.

`config.yaml`'s `model.id` (default `Qwen/Qwen2.5-32B-Instruct:featherless-ai`) and `model.hf_base_url` are set up for the HuggingFace Inference Router, but `StoryGenerator.load_model()` currently ignores `hf_base_url` and loads `model_id` as a local `transformers` checkpoint instead (see note above). Swapping in a different router-style model ID will not change behavior until the routed-inference code path is implemented.

---

## Project Structure

```
story_generator/
├── src/
│   ├── generator.py      # Core StoryGenerator class (all HF API calls)
│   ├── config.py         # Config dataclasses + YAML/env loader
│   └── save_outputs.py   # Structures and writes all output files
├── scripts/
│   └── run.py            # Local entrypoint (loads Qwen2.5-32B locally — needs a GPU)
├── cloud/
│   └── run.py            # Modal cloud entrypoint
├── API/
│   └── API.py            # Modal-hosted FastAPI endpoint
├── config.yaml           # All runtime settings
├── lesson.txt            # Your input lesson (edit this)
├── 3YNO.png              # Fixed host-character image, required for API deploys — bundled into every output zip
├── requirements.txt
├── Makefile
└── .env.example
```

---

## Requirements

- Python 3.11+
- A [HuggingFace](https://huggingface.co) account with a valid API token (`HF_TOKEN`)
- For cloud execution: a [Modal](https://modal.com) account
- A CUDA GPU (the current implementation loads Qwen2.5-32B-Instruct locally via `transformers`; the cloud/API paths request an H100 on Modal). `config.yaml` includes router-style settings (`model.hf_base_url`) for a future hosted-inference path, but that path isn't wired up yet.

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

Runs the full pipeline locally. This downloads and runs Qwen2.5-32B-Instruct on your machine via `transformers` — a CUDA GPU with enough VRAM for a 32B model is required.

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

The job runs as a Modal task in the cloud. Modal handles the container, secrets, and result download. `cloud/run.py` provisions an **H100 GPU** to load and run the model.

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
  dir: "output"

modal:
  app_name: "kids-story-generator"
  timeout: 3600
  python_version: "3.11"
```

`model.id` is written in router format (`repo:provider`) and is only actually consumed by `scripts/run.py`, which passes it straight into a local `transformers.from_pretrained()` call — the `:provider` suffix is not valid there and will likely cause the local run to fail to find the repo. `cloud/run.py` and `API/API.py` don't read `model.id` from `config.yaml` at all; they hardcode `MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"` directly in the file. To use a different model, edit `MODEL_ID` in `cloud/run.py` / `API/API.py` (for cloud/API runs) and set `model.id` in `config.yaml` to a plain HuggingFace repo id, without a `:provider` suffix (for local runs).

---

## Output Files

| File | Description |
|---|---|
| `characters.json` | List of characters: `name`, `description` (visual description), `output_path` (`characters/<name>.png`). Note: the fixed 3YNO host character is intentionally *not* included here — its image is copied in directly from `3YNO.png`. |
| `backgrounds.json` | List of backgrounds: `name`, `description`, `output_path` (`backgrounds/<name>.png`). |
| `voices.json` | Every voice line to synthesize — regular scene dialogue plus 3YNO host lines (API path only) — with `text`, `description` (voice description), and `output_path` (`voices/<name>.wav`). |
| `shots_flow.json` | The full ordered scene/shot manifest: `{"scenes": [...]}`, where each scene has `scene_id`, `title`, `background`, `is_host_scene`, `characters` (name/path/position), and `shots` (voice path, video prompt, negative prompt, speaker per shot). This is the file the `video_generator` module expects inside its input zip. |
| `characters/`, `backgrounds/`, `voices/` | Empty folders created alongside the JSON files, ready to receive the generated character/background images and voice audio from downstream models. |

---

## Notes

- The model performs up to **5 retry attempts** on any JSON-producing prompt before raising an error.
- Each dialogue line in the voice script is constrained to **15–20 words** for TTS pacing.
- Character positions are expressed as `(x, y)` floats from `(0.0, 0.0)` top-left to `(1.0, 1.0)` bottom-right, ready for compositing.
- Characters are always non-human (objects or nature elements) to keep the story grounded in the lesson topic.