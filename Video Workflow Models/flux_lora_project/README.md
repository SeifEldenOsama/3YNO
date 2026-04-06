# FLUX LoRA Fine-tuning

A clean, professional project for fine-tuning **FLUX.1-dev** with LoRA on Modal cloud (H100) or locally.

---

## 📁 Project Structure

```
flux-lora/
├── config.yaml               ← All settings in one place
├── requirements.txt
├── Makefile                  ← Shortcut commands
│
├── src/
│   ├── config.py             ← Config loader & dataclasses
│   ├── dataset.py            ← Kaggle / HuggingFace / Local dataset
│   ├── trainer.py            ← Full FLUX LoRA training loop
│   ├── inference.py          ← Image generation with trained LoRA
│   └── uploader.py           ← HuggingFace Hub upload / download
│
├── cloud/
│   ├── train.py              ← Remote training function (Modal H100)
│   └── inference.py          ← Remote inference function (Modal)
│
├── scripts/
│   ├── train.py              ← Local training CLI
│   ├── inference.py          ← Local inference CLI
│   └── upload.py             ← HF Hub upload/download CLI
│
└── outputs/                  ← Generated images + checkpoints (local)
```

---

## ⚡ Quick Start

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/flux-lora.git
cd flux-lora
pip install -r requirements.txt
```

### 2. Set up your secrets
```bash
cp .env.example .env
```
Then open `.env` and fill in your real values:
```env
HF_TOKEN=hf_your_token_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```
> ⚠️ **Never commit `.env` to git** — it's already in `.gitignore`

### 3. Edit config.yaml (non-secret settings only)
```yaml
hub:
  repo_id: "your_username/flux-lora-characters"

dataset:
  kaggle_dataset: "seifosamahosney/character-descriptions"

training:
  max_steps: 1000
  resolution: 512
```

### 4. Train on Modal (H100)

**Authenticate Modal:**
```bash
modal token set --token-id YOUR_ID --token-secret YOUR_SECRET
```

```bash
make modal-train
# or:
modal run cloud/train.py
```

### 5. Run inference on Modal
```bash
make modal-inference
# with custom prompt:
modal run cloud/inference.py --prompt "a warrior character" --num-images 4
```

### 6. Download results from Modal volume
```bash
modal volume get flux-lora-vol inference_outputs ./outputs/inference
```

### 7. Upload to HuggingFace
```bash
make upload
# or:
python scripts/upload.py
```

---

## 8. Run API

```bash
modal deploy API/API.py
```

---

## 🎛️ Configuration

All settings live in `config.yaml`. Key sections:

| Section | What it controls |
|---------|-----------------|
| `credentials` | HF token, Kaggle credentials |
| `dataset` | Source (kaggle/huggingface/local), dataset name |
| `lora` | Rank, alpha, target modules |
| `training` | Steps, batch size, learning rate, scheduler |
| `checkpointing` | Save frequency, output paths |
| `inference` | Prompt, steps, guidance scale, LoRA scale |
| `hub` | Repo ID, private/public |
| `modal` | GPU type, timeout, image config |

---

## 🖥️ CLI Reference

### Training
```bash
python scripts/train.py
python scripts/train.py --steps 500
python scripts/train.py --lr 5e-5 --rank 32
python scripts/train.py --output ./my_output
```

### Inference
```bash
python scripts/inference.py
python scripts/inference.py --prompt "a warrior" --num-images 4
python scripts/inference.py --lora-path ./outputs/checkpoint-500
python scripts/inference.py --seed 123 --cfg-scale 4.0
```

### Upload / Download
```bash
# Upload to HF
python scripts/upload.py
python scripts/upload.py --path ./outputs --repo username/my-lora

# Download from HF
python scripts/upload.py --download
python scripts/upload.py --download --path ./downloaded_lora
```

### Modal
```bash
# Train
modal run cloud/train.py

# Inference
modal run cloud/inference.py
modal run cloud/inference.py --prompt "a hero" --num-images 8 --seed 0
```

---

## 📊 Dataset Sources

Set `dataset.source` in `config.yaml`:

| Source | Config key | Description |
|--------|-----------|-------------|
| `kaggle` | `dataset.kaggle_dataset` | Downloads from Kaggle, pairs images with `.txt` captions |
| `huggingface` | `dataset.hf_dataset` | Streams from HF Hub dataset |
| `local` | `dataset.local_path` | Uses a local folder with images + `.txt` captions |

---

## 🔧 Technical Details

- **Model**: FLUX.1-dev (12B transformer-based diffusion model)
- **Training**: Flow matching with sigmoid timestep sampling
- **Precision**: bfloat16 throughout (no GradScaler needed)
- **LoRA**: Applied to attention projections (`to_q`, `to_k`, `to_v`, etc.)
- **Packing**: Latents packed to (B, H/2·W/2, C·4) as required by FLUX
- **Guidance**: Guidance tensor hardcoded at 3.5 (FLUX.1-dev distillation requirement)
