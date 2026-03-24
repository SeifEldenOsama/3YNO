# Harmony TTS

Full fine-tuning of [parler-tts/parler-tts-mini-v1](https://huggingface.co/parler-tts/parler-tts-mini-v1) for educational character voice generation, trained on Modal cloud (H100).

---

## Project Structure

```
harmony_tts/
├── config.yaml          ← all settings
├── .env                 ← credentials (never commit)
├── .env.example
├── .gitignore
├── requirements.txt
├── Makefile
│
├── src/
│   ├── config.py        ← config loader
│   ├── trainer.py       ← full fine-tuning training loop
│   ├── inference.py     ← audio generation
│   └── uploader.py      ← HF Hub upload/download
│
├── cloud/
│   ├── train.py         ← Modal training (H100)
│   └── inference.py     ← Modal inference
│
└── scripts/
    ├── train.py         ← local training CLI
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
HF_TOKEN=your_token_here
```

---

## Run on Modal

```bash
modal token set --token-id YOUR_ID --token-secret YOUR_SECRET
```

```bash
modal run cloud/train.py
```

```bash
modal run cloud/inference.py --text "Hello, this is Harmony speaking." --description "A female speaker delivers a cheerful and clear speech."
```

```bash
modal volume get tts-dataset-storage harmony-tts-output ./outputs/model
```

---

## Run Locally

```bash
python scripts/train.py
python scripts/inference.py --text "Hello" --description "A female speaker delivers a cheerful speech."
```

---

## Upload to HuggingFace

Set your repo in `config.yaml`:
```yaml
hub:
  repo_id: "your_username/Harmony_Parler_TTS"
```

```bash
python scripts/upload.py --path ./outputs/model
```

---

## Run API

```bash
modal deploy API/API.py
```

---

## Configuration

| Section | What it controls |
|---|---|
| `dataset` | HF dataset repo, column names, sample counts |
| `model` | Base model, tokenizers |
| `training` | Steps, batch size, learning rate, scheduler |
| `hub` | HF repo, private/public |
| `modal` | GPU type, timeout |

---

## Model

| | |
|---|---|
| Base model | `parler-tts/parler-tts-mini-v1` |
| Fine-tuning | Full fine-tuning (all weights) |
| Dataset | `SeifElden2342532/parler-tts-dataset-format` |
| Train samples | 18,700 |
| Eval samples | 2,000 |
| Max steps | 1,000 |
| Learning rate | 1e-5 (cosine) |
| GPU | H100 80GB |

---

## License

Apache 2.0
