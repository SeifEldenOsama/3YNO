# Video Generator

Audio-to-video generation using **LTX-2-19b** with Camera Control LoRA, run on Modal cloud (H200). Takes a character image and a voice audio file and produces an animated video clip with the character speaking in sync.

---

## Project Structure

```
video_generator/
├── config.yaml          ← all settings
├── .env                 ← credentials (never commit)
├── .env.example
├── .gitignore
├── requirements.txt
├── Makefile
│
├── src/
│   ├── config.py        ← config loader & dataclasses
│   └── generator.py     ← full generation pipeline (image prep, audio processing, video encoding)
│
├── cloud/
│   └── generate.py      ← Modal generation (H200)
│
└── scripts/
    └── generate.py      ← local generation CLI
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
HF_TOKEN=your_token_here
```

> ⚠️ **Never commit `.env` to git** — it's already in `.gitignore`

---

## Run on Modal

**Authenticate Modal:**
```bash
modal token set --token-id YOUR_ID --token-secret YOUR_SECRET
```

```bash
modal volume create ltx2-model-cache
```

**Generate a video:**
```bash
modal run cloud/generate.py --image-path test/droplet.png --audio-path test/surprised1.wav
```

**With a custom prompt and output path:**
```bash
modal run cloud/generate.py \
  --image-path test/droplet.png \
  --audio-path test/surprised1.wav \
  --prompt "A cartoon character speaking expressively" \
  --output-path output.mp4 \
  --seed 42
```

**Download results from Modal volume:**
```bash
modal volume get ltx2-model-cache output_hq.mp4 ./output_hq.mp4
```

---

## Run Locally

```bash
python scripts/generate.py --image test/droplet.png --audio test/surprised1.wav
python scripts/generate.py --image frame.png --audio voice.wav --output result.mp4
python scripts/generate.py --image frame.png --audio voice.wav --prompt "your prompt" --seed 0
```

---

## Makefile Shortcuts

```bash
make install
make generate          PROMPT="your prompt here"
make modal-generate    PROMPT="your prompt here"
```

---

## Configuration

| Section | What it controls |
|---|---|
| `model.id` | Base LTX-2 model ID on HuggingFace |
| `model.pipeline` | Custom audio-to-video pipeline |
| `model.lora_id` | Camera Control LoRA weights |
| `model.lora_scale` | LoRA blend strength (default 0.8) |
| `model.cache_dir` | Where model weights are cached on the volume |
| `generation.fps` | Output video frame rate (default 24) |
| `generation.num_steps` | Diffusion inference steps (default 8) |
| `generation.guidance_scale` | Classifier-free guidance (default 1.0) |
| `negative_prompt` | What to suppress in generation |

---

## Model

| | |
|---|---|
| Base model | `rootonchair/LTX-2-19b-distilled` |
| Pipeline | `multimodalart/ltx2-audio-to-video` |
| LoRA | `Lightricks/LTX-2-19b-LoRA-Camera-Control-Static` |
| LoRA scale | 0.8 |
| GPU | H200 80GB |
| Inference steps | 8 (distilled sigmas) |
| Supported resolutions | 512×512 · 768×512 · 512×768 (auto-selected from image ratio) |

---

## License

Apache 2.0
