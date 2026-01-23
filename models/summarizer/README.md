# Summarizer Module

This module provides a robust implementation for text summarization using the **Longformer Encoder-Decoder (LED)** architecture. It is optimized for processing long documents that exceed the token limits of standard transformer models.

## Features

- **Long Document Support**: Utilizes `allenai/led-base-16384` to handle up to 16,384 tokens.
- **End-to-End Pipeline**: Includes scripts for data preparation, training, and inference.
- **Configurable**: Easily adjust hyperparameters and model settings via `src/config.py`.

## Directory Structure

- `src/`: Core source code for the model and training.
- `notebooks/`: Jupyter notebooks for experimentation and demonstration.
- `data/`: Directory for storing training and evaluation datasets.
- `agent/`: Contains workflow diagrams and agent-related configurations.

## Usage

### Training
To train the model on your dataset, ensure your data is in the `data/` directory and run:
```bash
python src/train.py
```

### Inference
To generate a summary for a given text:
```bash
python src/generate.py --text "Your long text here"
```

## Configuration
Key parameters such as `MAX_INPUT_LENGTH`, `MAX_TARGET_LENGTH`, and `MODEL_CHECKPOINT` can be modified in `src/config.py`.
