# BART Summarizer

LoRA fine-tuning of [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) for abstractive summarization of educational lesson texts, trained on Modal cloud (H100).

---

## Project Structure

```
led_summarizer/
├── config.yaml          ← all settings
├── .env                 ← credentials (never commit)
├── .env.example
├── .gitignore
├── requirements.txt
├── Makefile
│
├── src/
│   ├── config.py        ← config loader
│   ├── dataset.py       ← CSV loading + tokenization
│   ├── trainer.py       ← LoRA training loop
│   ├── inference.py     ← summarization inference
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
modal volume create led-summarizer-vol
modal volume put led-summarizer-vol data_summarization.csv /data_summarization.csv
```

```bash
modal run cloud/train.py
```

```bash
modal run cloud/inference.py --text "your lesson text here"
```

```bash
modal volume get led-summarizer-vol bart-lora-output ./outputs/model
```

---

## Run Locally

```bash
python scripts/train.py --csv data_summarization.csv
python scripts/inference.py --text "your lesson text here"
python scripts/inference.py --csv data.csv
```

---

## Upload to HuggingFace

Set your repo in `config.yaml`:
```yaml
hub:
  repo_id: "your_username/bart-summarizer"
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
| `dataset` | CSV path, column names, splits |
| `model` | Base model, max input/output length |
| `lora` | LoRA rank, alpha, dropout, target modules |
| `training` | Epochs, batch size, learning rate |
| `inference` | Beam size, max length |
| `hub` | HF repo, private/public |
| `modal` | GPU type, timeout |

---

## Model

| | |
|---|---|
| Base model | `facebook/bart-large-cnn` |
| Fine-tuning | LoRA (rank 16, alpha 32) |
| Trainable params | ~5M out of 406M (~1.2%) |
| Task | Abstractive summarization |
| Max input | 1024 tokens |
| Max output | 256 tokens |
| Training | 8 epochs on H100 80GB |
| Adapter size | ~33 MB |

---

## License

Apache 2.0
