# Story Generator Module

The Story Generator module leverages the power of **Mistral-7B** to create engaging and coherent stories. It uses a multi-stage pipeline to ensure the generated content follows a logical progression from premise to final narrative.

## Features

- **Quantized Inference**: Uses 4-bit quantization (bitsandbytes) to run the Mistral-7B model efficiently on consumer hardware.
- **Structured Pipeline**:
  1. **Premise Generation**: Creates the initial story idea.
  2. **Planning**: Outlines the story structure.
  3. **Story Execution**: Generates the full narrative based on the plan.
- **Robust Client**: Includes an `LLMClient` with built-in retry logic and post-processing capabilities.

## Directory Structure

- `models/`: Contains the LLM client implementation.
- `pipeline/`: Scripts for the different stages of story generation.
- `utils/`: Utility functions and configuration.
- `notebooks/`: Demonstration of the story generation process.

## Installation

Install the specific dependencies for this module:
```bash
pip install -r requirements.txt
```

## Usage

You can run the story generation pipeline through the provided notebook `notebooks/final-story-generation.ipynb` or by integrating the pipeline scripts into your application.

```python
from pipeline.story import StoryGenerator
# Initialize and generate
```

## Requirements

- `transformers`
- `bitsandbytes`
- `accelerate`
- `langchain`
- `torch`
