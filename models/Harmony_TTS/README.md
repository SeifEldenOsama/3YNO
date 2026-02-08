# 🗣️ Parler-TTS Fine-Tuning on Modal (H100 Optimized)

## Project Overview 

This repository contains the infrastructure and scripts for fine-tuning the **Parler-TTS** model, a state-of-the-art Text-to-Speech (TTS) model from Hugging Face, to generate high-fidelity audio in a specific voice or style.

The entire workflow is designed for **ultra-optimized performance** and **reproducibility** using the [Modal](https://modal.com/) serverless cloud platform, specifically leveraging the immense power of the **NVIDIA H100 GPU** for accelerated training.

## Key Features

| Feature | Description | Benefit |
| :--- | :--- | :--- |
| **Parler-TTS Fine-Tuning** | Adapts the base Parler-TTS model to a custom voice dataset. | Generates high-quality, personalized speech. |
| **Modal Integration** | Scripts are written for seamless execution on the Modal platform. | Serverless, scalable, and reproducible cloud execution. |
| **NVIDIA H100 Optimization** | Training environment is configured to utilize the powerful H100 GPU. | Achieves significantly faster training times and handles larger models. |
| **Modularized Scripts** | Core functionalities are separated into standalone Python scripts for easier execution and maintenance. | Improves workflow clarity and professional standards. |

## Dataset

The fine-tuning process relies on a custom dataset. For this specific run, the model was **trained on 9,000 samples** and **evaluated on 500 samples**. The data preparation involves two main components: the raw audio data and the processed Hugging Face dataset repository.

| Component | Link |
| :--- | :--- |
| **Raw Audio Data (Kaggle)** | [https://www.kaggle.com/datasets/seifosamahosney/tts-dataset](https://www.kaggle.com/datasets/seifosamahosney/tts-dataset) |
| **Hugging Face Dataset Repo** | `https://huggingface.co/datasets/SeifElden2342532/parler-tts-dataset-format` |
| **Hugging Face Model Repo** | `https://huggingface.co/SeifElden2342532/Harmony_Parler_TTS` |

## Repository Structure

The project is structured as follows:

```
.
├── notebooks/
│   ├── TTS_Gemeni_Style-fine-tuned_v2.ipynb # Latest Modal notebook (updated)
│   └── comparison_test/                    # Sample audio comparisons
├── scripts/
│   ├── download_tts_simple.py               # Modal script to download and structure data from Kaggle
│   ├── train_parler.py                      # Modal script for H100-optimized Parler-TTS fine-tuning
│   └── compare_models.py                    # Modal script for running comparison between base and fine-tuned models
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

## Usage with Modal

### Prerequisites

1.  Install the Modal CLI: `pip install modal-client`
2.  Log in to Modal: `modal login`
3.  Create a Modal Secret named `kaggle-secret` with your Kaggle credentials (`KAGGLE_USERNAME` and `KAGGLE_KEY`).

### 1. Data Preparation

The `download_tts_simple.py` script handles downloading the Kaggle dataset, unzipping it, and structuring it correctly into a persistent Modal Volume named `tts-dataset-storage`.

```bash
modal run scripts/download_tts_simple.py
```

### 2. Model Fine-Tuning (H100)

The `train_parler.py` script executes the fine-tuning process. It is configured to request a dedicated **NVIDIA H100 GPU** instance on Modal for maximum training efficiency.

```bash
modal run scripts/train_parler.py
```

### 3. Model Comparison

The `compare_models.py` script allows you to generate audio from both the base model and your fine-tuned model to compare the results.

```bash
modal run scripts/compare_models.py
```
