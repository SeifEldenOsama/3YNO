<p align="center">
  <img src="assets/logo.png" alt="3YNO Logo" width="200">
</p>

# 3YNO: AI-Powered Educational Suite for Dyslexics & Visual Learners

> **Status: Under Development** 🚧
> 3YNO is an innovative educational platform designed to bridge the learning gap for dyslexic individuals and visual learners by transforming complex text into engaging, story-driven visual content.

---

## Project Vision

3YNO (pronounced "EYE-NO") addresses the challenges faced by learners who struggle with traditional text-heavy educational materials. By leveraging state-of-the-art Artificial Intelligence, the application automates the conversion of books, research papers, and general text into explanatory videos.

### Core Workflow

The application follows a sophisticated multi-stage pipeline to ensure educational content is both accurate and accessible:

1. **Content Extraction**: Uploaded text is analyzed to extract core scientific and educational concepts.
2. **Narrative Transformation**: Extracted content is reframed into a compelling story or narrative structure, making it easier to digest.
3. **Character Creation**: AI-driven generation of characters that guide the learner through the narrative.
4. **Voice Generation**: AI-driven synthesis of character voices using fine-tuned Text-to-Speech models.
5. **Video Synthesis**: (In Development) Final conversion of the narrative and characters into an explanatory video.

---

## Repository Structure

The project is currently organized into specialized modules that handle different stages of the pipeline:

| Module | Role in Pipeline | Model | Status |
| :--- | :--- | :--- | :--- |
| **[Summarizer](./models/summarizer)** | Content Extraction & Scientific Distillation | LED-base-16384 (fine-tuned) | Active Development |
| **[Story Generator](./models/story_generator)** | Narrative Transformation & Character Planning | Mistral-7B | Active Development |
| **[Character Generator](./models/character_generator)** | Character & Background Image Generation | FLUX.1-dev (LoRA fine-tuned) | Active Development |
| **[Harmony TTS](./models/Harmony_TTS)** | Voice Generation for Characters | Custom TTS | Active Development |

---

## Models

### Summarizer — LED fine-tuned
Fine-tuned [allenai/led-base-16384](https://huggingface.co/allenai/led-base-16384) for abstractive summarization of educational texts.

| | |
|---|---|
| Base model | `allenai/led-base-16384` |
| Dataset | Custom lesson descriptions (~2000 samples) |
| Max input | 1024 tokens |
| Max output | 256 tokens |
| Training | 8 epochs on H100 80GB |
| HF Repo | [seifosamahosney/led-summarizer](https://huggingface.co/seifosamahosney/led-summarizer) |

---

### Character Generator — FLUX.1-dev LoRA
Fine-tuned [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) with LoRA for generating educational characters and visual scenes.

| | |
|---|---|
| Base model | `black-forest-labs/FLUX.1-dev` |
| Method | LoRA (rank 16, alpha 16) |
| Dataset | Character descriptions (~2016 images) |
| Training | 2000 steps on H100 80GB |
| HF Repo | [seifosamahosney/flux-lora-characters](https://huggingface.co/seifosamahosney/flux-lora-characters) |

---

## Getting Started

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended for local model inference)

### Installation
```bash
git clone https://github.com/SeifEldenOsama/3YNO.git
cd 3YNO
```

Refer to individual module READMEs for specific dependency installation.

---

## Roadmap

- [x] Initial Summarization Pipeline (LED-based)
- [x] Narrative Generation Framework (Mistral-7B)
- [x] Character & Background Image Generation (FLUX.1-dev LoRA)
- [x] Voice Generation for each character
- [ ] Apply each voice generated for each character with its image, to generate small video scene for each one
- [ ] Integrate all the video scenes into one educational interactive video
