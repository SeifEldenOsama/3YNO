# Harmony TTS

Voice generation for educational characters powered by the **Google Gemini TTS API** (`gemini-2.5-flash-preview-tts`), run on Modal cloud (serverless). Supports parallel multi-voice generation with automatic API-key rotation and fallback.


---

## Project Structure

```
Harmony_TTS/
├── config.yaml          ← settings (model, modal)
├── .env                 ← credentials (never commit)
├── .env.example
├── .gitignore
├── requirements.txt
├── Makefile
│
├── src/
│   ├── config.py        ← config loader
│   ├── trainer.py       ← (legacy — not used with Gemini TTS)
│   ├── inference.py     ← GeminiTTSInference class, key rotation, parallel generation
│   └── uploader.py      ← HF Hub upload/download (for other assets)
│
├── cloud/
│   ├── inference.py     ← Modal entrypoint (Gemini TTS)
│   └── train.py         ← (legacy — not used with Gemini TTS)
│
└── scripts/
    ├── train.py         ← (legacy)
    ├── inference.py     ← local inference CLI
    └── upload.py        ← HF Hub CLI
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
GEMINI_API_KEYS=your_key_1,your_key_2,your_key_3
```

Supply one key or several comma-separated keys. Multiple keys enable parallel generation and automatic fallback rotation on quota errors.

> ⚠️ **Never commit `.env` to git** — it's already in `.gitignore`

---

## Run on Modal

```bash
modal token set --token-id YOUR_ID --token-secret YOUR_SECRET
```

```bash
modal run cloud/inference.py --text "Hello, this is Harmony speaking." --description "Aoede A calm and friendly female voice with a warm clear tone." --output output.wav
```

The audio is saved locally at the path given by `--output`.

---

## Run Locally

```bash
python scripts/inference.py --text "Hello" --description "Aoede A calm and friendly female voice with a warm clear tone."
```

---

## Run API

```bash
modal deploy API/API.py
```

---

## Available Voices

The voice name is always the **first word** of the `description` argument. The rest of the description is used as context only.

| Voice | Gender |
|---|---|
| `Puck` | Male |
| `Charon` | Male |
| `Orus` | Male |
| `Achird` | Male |
| `Enceladus` | Male |
| `Zephyr` | Female |
| `Leda` | Female |
| `Kore` | Female |
| `Aoede` | Female |
| `Gacrux` | Female |
| `Sulafat` | Female |

Example description: `"Aoede A warm and expressive female voice, slow and clear."`

---

## Parallel Generation

`src/inference.py` exposes a `generate_parallel` function that takes a list of `{text, description}` dicts and generates all clips concurrently, each using a different API key:

```python
from src.inference import generate_parallel

clips = generate_parallel([
    {"text": "Hello!", "description": "Aoede A cheerful female voice."},
    {"text": "Welcome!", "description": "Puck A friendly male voice."},
])
# clips → list of WAV bytes, in the same order as input
```

---

## Configuration

| Section | What it controls |
|---|---|
| `model` | Base model, tokenizers |
| `training` | (legacy — not applicable to Gemini TTS) |
| `hub` | HF repo, private/public |
| `modal` | Timeout, memory |

---

## Model

| | |
|---|---|
| Provider | Google Gemini TTS API |
| Model | `gemini-2.5-flash-preview-tts` |
| Approach | API inference (no fine-tuning, no GPU) |
| Key management | `GEMINI_API_KEYS` (comma-separated, with fallback rotation) |
| Output format | WAV (24 kHz, mono, 16-bit PCM) |
| GPU required | No |
| Cloud runtime | [Modal](https://modal.com) |

---

## Huggingface space
https://huggingface.co/spaces/SeifElden2342532/Harmony-tts

## License

Apache 2.0
