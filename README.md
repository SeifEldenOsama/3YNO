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
5. **Video Synthesis**: Final conversion of the narrative and characters into an explanatory video.

---

## Repository Structure

| Module | Role in Pipeline | Model | Approach |
| :--- | :--- | :--- | :--- |
| **[Summarizer](./Video%20Workflow%20Models/bart_summarizer)** | Content Extraction & Scientific Distillation | BART-large-CNN | LoRA fine-tuning |
| **[Story Generator](./Video%20Workflow%20Models/story_generator)** | Narrative Transformation & Character Planning | Qwen2.5-32B-Instruct | Prompt Engineering |
| **[Character Generator](./Video%20Workflow%20Models/flux_lora_project)** | Character & Background Image Generation | FLUX.1-dev | LoRA fine-tuning |
| **[Harmony TTS](./Video%20Workflow%20Models/Harmony_TTS)** | Voice Generation for Characters | Gemini 2.5 Flash TTS | API inference (prompt engineering) |
| **[Video Generator](./Video%20Workflow%20Models/video_generator)** | Character Video Synthesis | LTX-2-19b-distilled | Audio-to-video + Camera Control |
| **[3YNO Chatbot](./3YNO%20Chatbot)** | Dyslexia Support AI Assistant | Gemini 2.5 Flash | Prompt Engineering |
| **[Quiz Generator](./Quiz%20Generator)** | Post-Video MCQ Generation & Evaluation | Llama 3.1 8B (via Groq) | Prompt Engineering |
| **[OCR](./OCR)** | Document/Camera Text Extraction | GLM-OCR (zai-org/GLM-OCR) | Inference only |

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

Uses [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) via prompt engineering to transform educational lessons into structured children's stories with characters, scenes, and voice-ready scripts. No fine-tuning required — the model follows detailed multi-stage prompts out of the box. The model is currently loaded and run directly on a GPU worker (not called through a hosted inference API); see the module README for details.

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-32B-Instruct` |
| Approach | Prompt engineering (no fine-tuning) |
| GPU | H100 |
| Cloud runtime | [Modal](https://modal.com) |
| Module path | [`./Video Workflow Models/story_generator`](./Video%20Workflow%20Models/story_generator) |

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
| Module path | [`./Video Workflow Models/video_generator`](./Video%20Workflow%20Models/video_generator) |

---

### Harmony TTS — Gemini TTS (API inference)

Voice generation for educational characters, powered by the **Google Gemini TTS API** (`gemini-2.5-flash-preview-tts`) with automatic API-key rotation and fallback. The module also contains a legacy full fine-tuning pipeline for `parler-tts/parler-tts-mini-v1`, which is kept in the codebase but is **not used** by the current inference path.

| | |
|---|---|
| Provider | Google Gemini TTS API |
| Model | `gemini-2.5-flash-preview-tts` |
| Approach | API inference (prompt engineering), no fine-tuning, no GPU |
| Legacy (unused) | Full fine-tuning of `parler-tts/parler-tts-mini-v1` — see module README |
| Cloud runtime | [Modal](https://modal.com) |
| Module path | [`./Video Workflow Models/Harmony_TTS`](./Video%20Workflow%20Models/Harmony_TTS) |

---

### 3YNO Chatbot — Gemini 2.5 Flash

A conversational AI assistant specialised in dyslexia support and visual learning. Powered by Google Gemini 2.5 Flash and deployed on Modal with a FastAPI endpoint. Provides parents, teachers, and caregivers with expert guidance on dyslexia signs, strategies, and resources.

| | |
|---|---|
| Model | `gemini-2.5-flash` |
| Approach | Prompt engineering (system prompt) |
| API | Google Generative AI |
| Cloud runtime | [Modal](https://modal.com) |
| Module path | [`./3YNO Chatbot`](./3YNO%20Chatbot) |

---

### Quiz Generator — Llama 3.1 8B (via Groq)

Generates 5 multiple-choice questions from a summary/lesson text and evaluates submitted answers. Built with FastAPI + LangChain, deployed on Modal.

| | |
|---|---|
| Model | `llama-3.1-8b-instant` |
| Provider | Groq API (via LangChain) |
| Approach | Prompt engineering, no fine-tuning |
| Cloud runtime | [Modal](https://modal.com) |
| Module path | [`./Quiz Generator`](./Quiz%20Generator) |

---

### OCR — GLM-OCR

Document/camera image understanding and text extraction using **zai-org/GLM-OCR**, supporting text, formula, table, and structured-JSON recognition.

| | |
|---|---|
| Model | `zai-org/GLM-OCR` |
| Approach | Inference only, no fine-tuning |
| GPU | A10G |
| Cloud runtime | [Modal](https://modal.com) |
| Module path | [`./OCR`](./OCR) |

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
- [x] Integrate all the video scenes into one educational interactive video
- [x] 3YNO Chatbot — dyslexia support AI assistant (Gemini 2.5 Flash)
- [x] Quiz Generator after each video
- [x] OCR text reader for camera upload