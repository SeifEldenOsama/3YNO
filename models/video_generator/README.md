# Video Generator

Audio-to-video generation using **LTX-2-19b** with Camera Control LoRA, run on Modal cloud (H200). Takes a background image, one or more character PNGs (white background), and a voice audio file — composites them into a single frame, then generates an animated video clip with the characters speaking in sync.

---

## Project structure

```
video_generator/
├── config.yaml              ← all settings
├── .env                     ← credentials (never commit)
├── .env.example
├── .gitignore
├── requirements.txt
├── Makefile
│
├── src/
│   ├── config.py            ← config loader & dataclasses
│   ├── compositor.py        ← white-bg removal, multi-character compositing, last-frame extraction
│   ├── generator.py         ← image/audio prep, LTX-2 inference, video encoding, pipeline loop
│   └── video_utils.py       ← resolution helpers, frame/audio utilities
│
├── cloud/
│   └── generate.py          ← Modal app — single shot (main) and full pipeline (run_pipeline)
│
└── scripts/
    └── generate.py          ← local CLI for single shot
```

---

## Setup

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Fill in `.env`:
```env
HF_TOKEN=your_huggingface_token_here
```

> ⚠️ **Never commit `.env` to git** — it's already in `.gitignore`

**Authenticate Modal:**
```bash
modal token set --token-id YOUR_TOKEN_ID --token-secret YOUR_TOKEN_SECRET
```

**Create the model cache volume (first time only):**
```bash
modal volume create ltx2-model-cache
```

> On first run Modal downloads the LTX-2 model (~20 GB) into the volume. This takes ~5–10 minutes. Every subsequent run skips the download.

---

## Run on Modal

### Single shot

Supply one pre-composited frame PNG + one audio file:

```bash
modal run cloud/generate.py::main --image-path droplet.png --audio-path surprised1.wav --prompt "A water droplet character speaking expressively, smooth animation, high quality" --output-path outputs/test_shot.mp4
```

Saves two files: `outputs/test_shot.mp4` and `outputs/test_shot_last_frame.png`.

All arguments:

| Argument | Default | Description |
|---|---|---|
| `--image-path` | `test/frame.png` | Path to frame PNG |
| `--audio-path` | `test/audio.wav` | Path to audio WAV/MP3 |
| `--prompt` | required | LTX-2 video prompt |
| `--output-path` | `output.mp4` | Where to save the clip |
| `--seed` | `-1` (random) | Set a fixed seed for reproducibility |

### Full pipeline

Reads `video_timeline.json` and all assets from disk, runs all shots remotely, saves clips per shot.

```bash
modal run cloud/generate.py::run_pipeline --timeline-path outputs/video_timeline.json --assets-dir outputs --audio-dir outputs --output-dir outputs/clips
```

All arguments:

| Argument | Default | Description |
|---|---|---|
| `--timeline-path` | `outputs/video_timeline.json` | Path to the timeline JSON |
| `--assets-dir` | `outputs` | Root folder containing `assets/` and `scenes/` |
| `--audio-dir` | `outputs` | Root folder for audio files (same as assets-dir usually) |
| `--output-dir` | `outputs/clips` | Where to save the generated clips |
| `--seed` | `42` | Seed for all shots |

---

## Timeline JSON format

The full pipeline expects this layout on disk:

```
outputs/
  video_timeline.json
  assets/
    backgrounds/forest.png
    characters/hero.png
    characters/villain.png
  scenes/
    scene_01/shots/shot_01/voice.mp3
    scene_01/shots/shot_02/voice.mp3
```

Minimal `video_timeline.json`:

```json
{
  "shots": [
    {
      "shot_id":       "scene_01_shot_01",
      "scene_number":  1,
      "shot_number":   1,
      "background_name": "forest",
      "frame_source":  "composite",
      "video_prompt":  "Two characters in a forest, expressive animation, high quality",
      "voice_file":    "scenes/scene_01/shots/shot_01/voice.mp3",
      "characters_present": [
        {"name": "hero",    "position": {"x": 0.3, "y": 0.75}},
        {"name": "villain", "position": {"x": 0.7, "y": 0.75}}
      ]
    },
    {
      "shot_id":       "scene_01_shot_02",
      "scene_number":  1,
      "shot_number":   2,
      "background_name": "forest",
      "frame_source":  "previous_clip",
      "video_prompt":  "Character reacts with surprise, smooth animation",
      "voice_file":    "scenes/scene_01/shots/shot_02/voice.mp3",
      "characters_present": [
        {"name": "hero",    "position": {"x": 0.3, "y": 0.75}},
        {"name": "villain", "position": {"x": 0.7, "y": 0.75}}
      ]
    }
  ]
}
```

**`frame_source` values:**

| Value | Behaviour |
|---|---|
| `"composite"` | Composites background + characters into a fresh frame (shot 1 of each scene) |
| `"previous_clip"` | Extracts the last frame of the previous clip for continuity (shot 2, 3, 4…) |

---

## Multi-character compositing

`compositor.py` processes characters in list order onto the background:

1. Load background → resize to output resolution
2. For each character in `characters_present`:
   - Remove white/near-white background pixels (threshold R≥240 AND G≥240 AND B≥240)
   - Scale to 35% of canvas height
   - Place center at `(x × W, y × H)`
   - Paste onto canvas using alpha mask
3. Convert canvas to flat RGB PNG → pass to LTX-2

**Position reference (x, y from 0.0 to 1.0, center of character):**

```
(0,0) ──────────── (1,0)
  │                  │
  │   (0.3, 0.75)    │   ← good spot for left character
  │         (0.7, 0.75)   ← good spot for right character
(0,1) ──────────── (1,1)
```

**Notes:**
- Characters are painted in list order — the last character in the list appears on top if they overlap.
- All characters are scaled to the same fixed height (35% of canvas). There is no per-character size control.
- If character images have light-colored hair, white clothing, or bright highlights, lower the removal threshold in `compositor.py` from `240` to around `210–220` to avoid accidentally erasing parts of the character.

---

## Run locally

```bash
python scripts/generate.py \
  --frame  frame.png \
  --audio  voice.wav \
  --prompt "your prompt here" \
  --output output.mp4
```

Saves `output.mp4` and `output_last_frame.png`.

---

## Makefile shortcuts

```bash
make install
make generate        PROMPT="your prompt here"
make modal-generate  PROMPT="your prompt here"
```

---

## Configuration (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `model.id` | `rootonchair/LTX-2-19b-distilled` | Base model on HuggingFace |
| `model.pipeline` | `multimodalart/ltx2-audio-to-video` | Custom audio-to-video pipeline |
| `model.lora_id` | `Lightricks/LTX-2-19b-LoRA-Camera-Control-Static` | Camera Control LoRA |
| `model.lora_scale` | `0.8` | LoRA blend strength |
| `model.cache_dir` | `/model-cache` | Weight cache path on Modal volume |
| `generation.quality` | `fhd` | Resolution preset: `sd` / `hd` / `fhd` |
| `generation.fps` | `24.0` | Output frame rate |
| `generation.seed` | `-1` | `-1` = random each run |
| `generation.num_steps` | `8` | Diffusion steps (distilled schedule) |
| `generation.guidance_scale` | `1.0` | CFG guidance strength |
| `generation.bright_tail` | `1.6` | Seconds of visible settle after audio ends |
| `generation.dark_buffer` | `2.0` | Silent seconds trimmed before delivery |
| `negative_prompt` | see config | Suppresses text, camera movement, blur, etc. |
| `modal.gpu` | `H200` | GPU type |
| `modal.timeout` | `7200` | Max job duration in seconds |

---

## Model

| | |
|---|---|
| Base model | `rootonchair/LTX-2-19b-distilled` |
| Pipeline | `multimodalart/ltx2-audio-to-video` |
| LoRA | `Lightricks/LTX-2-19b-LoRA-Camera-Control-Static` |
| LoRA scale | `0.8` |
| GPU | H200 141 GB |
| Inference steps | 8 (distilled sigma schedule) |
| Supported resolutions | 512×512 · 768×512 · 512×768 (auto-selected from image aspect ratio) |

---

## License

Apache 2.0
