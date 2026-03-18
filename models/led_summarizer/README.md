# LED Summarizer

Fine-tuning [allenai/led-base-16384](https://huggingface.co/allenai/led-base-16384) for abstractive summarization of lesson/educational texts, trained on Modal cloud (H100).

---

## Project Structure

```
led_summarizer/
├── config.yaml              ← all settings
├── .env                     ← credentials (never commit)
├── .env.example
├── .gitignore
├── requirements.txt
├── Makefile
│── data_summarization.csv
│
│── API/
│   ├── API.py
│
├── src/
│   ├── config.py            ← config loader
│   ├── dataset.py           ← CSV loading + tokenization
│   ├── trainer.py           ← full LED training loop
│   ├── inference.py         ← summarization inference
│   └── uploader.py          ← HF Hub upload/download
│
├── cloud/
│   ├── train.py             ← Modal training (H100)
│   └── inference.py         ← Modal inference
│
└── scripts/
    ├── train.py             ← local training CLI
    ├── inference.py         ← local inference CLI
    └── upload.py            ← HF Hub CLI
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

**Authenticate Modal:**
```bash
modal token set --token-id YOUR_ID --token-secret YOUR_SECRET
```

**Upload your CSV:**
```bash
modal volume create led-summarizer-vol
modal volume put led-summarizer-vol data_summarization.csv /data_summarization.csv
```

**Train:**
```bash
modal run cloud/train.py
```

**Inference:**
```bash
modal run cloud/inference.py --text "your lesson text here"
```

**Download model:**
```bash
modal volume get led-summarizer-vol led-summarizer-output ./outputs/model
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
  repo_id: "your_username/led-summarizer"
```

Then:
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

All settings are in `config.yaml`:

| Section | What it controls |
|---|---|
| `dataset` | CSV path, column names, splits |
| `model` | Base model, max input/output length |
| `training` | Epochs, batch size, learning rate |
| `inference` | Beam size, max length |
| `hub` | HF repo, private/public |
| `modal` | GPU type, timeout |

---

## Model

| | |
|---|---|
| Base model | `allenai/led-base-16384` |
| Task | Abstractive summarization |
| Max input | 1024 tokens |
| Max output | 256 tokens |
| GPU | H100 80GB |

---

## License

Apache 2.0
