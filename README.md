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

| Module | Role in Pipeline | Model | Approach | Status |
| :--- | :--- | :--- | :--- | :--- |
| **[Summarizer](./models/bart_summarizer)** | Content Extraction & Scientific Distillation | BART-large-CNN | LoRA fine-tuning | Active Development |
| **[Story Generator](./models/story_generator)** | Narrative Transformation & Character Planning | Qwen2.5-32B-Instruct | Prompt Engineering | Active Development |
| **[Character Generator](./models/flux_lora_project)** | Character & Background Image Generation | FLUX.1-dev | LoRA fine-tuning | Active Development |
| **[Harmony TTS](./models/Harmony_TTS)** | Voice Generation for Characters | Parler-TTS-mini-v1 | Full fine-tuning | Active Development |
| **[Video Generator](./models/video_generator)** | Character Video Synthesis | LTX-2-19b-distilled | Audio-to-video + Camera Control LoRA | Active Development |

---

## Models

### Summarizer — BART LoRA fine-tuned

LoRA fine-tuning of [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) for abstractive summarization of educational texts. Only ~1.2% of parameters are trained, making it fast to train and lightweight to deploy.

| | |
|---|---|
| Base model | `facebook/bart-large-cnn` |
| Approach | LoRA fine-tuning (rank 16, alpha 32) |
| Trainable params | ~5M out of 406M (~1.2%) |
| Dataset | Custom lesson descriptions (~2000 samples) |
| Max input | 1024 tokens |
| Max output | 256 tokens |
| Training | 8 epochs on H100 80GB |
| HF Repo | [SeifElden2342532/children_educational_summarizer](https://huggingface.co/SeifElden2342532/children_educational_summarizer) |

---

### Story Generator — Qwen2.5-32B-Instruct

Uses [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) via prompt engineering to transform educational lessons into structured children's stories with characters, scenes, and voice-ready scripts. No fine-tuning required — the model follows detailed multi-stage prompts out of the box.

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-32B-Instruct` |
| Approach | Prompt engineering (no fine-tuning) |
| Quantization | 4-bit NF4 via `bitsandbytes` |
| GPU | A100 80GB |
| Cloud runtime | [Modal](https://modal.com) |
| Module path | [`./models/story_generator`](./models/story_generator) |

---

### Character Generator — FLUX.1-dev LoRA

Fine-tuned [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) with LoRA for generating educational characters and visual scenes.

| | |
|---|---|
| Base model | `black-forest-labs/FLUX.1-dev` |
| Approach | LoRA fine-tuning (rank 16, alpha 16) |
| Dataset | Character descriptions (~2016 images) |
| Training | 2000 steps on H100 80GB |
| HF Repo | [SeifElden2342532/flux-lora-characters](https://huggingface.co/SeifElden2342532/flux-lora-characters) |

---

### Video Generator — LTX-2 Audio-to-Video

Takes a character image and a voice audio clip produced by Harmony TTS, and synthesises an animated video of the character speaking. Uses [rootonchair/LTX-2-19b-distilled](https://huggingface.co/rootonchair/LTX-2-19b-distilled) with a Camera Control LoRA for stable, expressive output.

| | |
|---|---|
| Base model | `rootonchair/LTX-2-19b-distilled` |
| Pipeline | `multimodalart/ltx2-audio-to-video` |
| LoRA | `Lightricks/LTX-2-19b-LoRA-Camera-Control-Static` |
| Approach | Audio-to-video inference (no fine-tuning) |
| GPU | H200 141GB |
| Inference steps | 8 (distilled sigmas) |
| Module path | [`./models/video_generator`](./models/video_generator) |

---

### Harmony TTS — Parler-TTS full fine-tuned

Full fine-tuning of [parler-tts/parler-tts-mini-v1](https://huggingface.co/parler-tts/parler-tts-mini-v1) for generating expressive character voices in educational videos.

| | |
|---|---|
| Base model | `parler-tts/parler-tts-mini-v1` |
| Approach | Full fine-tuning (all weights) |
| Dataset | `SeifElden2342532/parler-tts-dataset-format` (18,700 samples) |
| Max steps | 1,000 |
| Learning rate | 1e-5 (cosine) |
| GPU | H100 80GB |
| HF Repo | [SeifElden2342532/Harmony_Parler_TTS](https://huggingface.co/SeifElden2342532/Harmony_Parler_TTS) |

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

- [x] Initial Summarization Pipeline (BART-large-CNN + LoRA)
- [x] Narrative Generation Framework (Qwen2.5-32B — prompt engineering)
- [x] Character & Background Image Generation (FLUX.1-dev LoRA)
- [x] Voice Generation for each character (Harmony TTS)
- [x] Apply each voice generated for each character with its image, to generate small video scene for each one
- [ ] Integrate all the video scenes into one educational interactive video
