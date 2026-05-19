# GLM-OCR Project

A clean, professional project for document understanding and OCR using the **zai-org/GLM-OCR** model. This project supports local execution, remote execution on Modal (GPUs), and deployment as a FastAPI service.

---

## 📁 Project Structure

```
glm-ocr-project/
├── config.yaml               ← Centralized configuration
├── requirements.txt          ← Dependencies (includes latest transformers)
├── Makefile                  ← Shortcut commands
│
├── src/
│   ├── config.py             ← Config loader
│   └── inference.py          ← Core GLM-OCR logic
│
├── cloud/
│   └── inference.py          ← Remote inference (Modal)
│
├── scripts/
│   └── inference.py          ← Local CLI for OCR
│
├── API/
│   └── API.py                ← FastAPI deployment on Modal
│
└── outputs/                  ← Local results
```

---

## ⚡ Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```
*Note: This project requires the development version of transformers:*
`pip install git+https://github.com/huggingface/transformers.git`

### 2. Local Inference
```bash
python scripts/inference.py --image path/to/your/image.png --prompt "Text Recognition:"
```

### 3. Remote Inference (Modal)
Ensure you have Modal configured (`modal token set`).
```bash
modal run cloud/inference.py --image-path path/to/your/image.png --prompt "Text Recognition:"
```

### 4. Deploy API
```bash
modal deploy API/API.py
```

---

## 🎛️ Configuration

Edit `config.yaml` to change model settings, GPU types, or default prompts.

| Section | Description |
|---------|-------------|
| `model` | Model ID and device settings |
| `inference` | Default prompt and token limits |
| `modal` | GPU type (e.g., A10G) and cache settings |

---

## 🔍 Supported Prompts

GLM-OCR works best with specific prompts:
- `Text Recognition:` - Standard OCR
- `Formula Recognition:` - For mathematical formulas
- `Table Recognition:` - For structured tables
- Custom JSON schemas for Information Extraction.
