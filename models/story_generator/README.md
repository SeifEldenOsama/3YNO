# Story Generator Module

The Story Generator module leverages **Mistral-7B** to transform educational summaries into engaging narratives for dyslexic and visual learners.

## Pipeline Integration

The module is designed as a sequential pipeline where each stage feeds into the next:

1.  **Premise (`pipeline/premise.py`)**: 
    - **Input**: Educational summary (distilled scientific content).
    - **Output**: `output/premise.json` (Title and Story Premise).
2.  **Plan (`pipeline/plan.py`)**: 
    - **Input**: `output/premise.json`.
    - **Output**: `output/plan.json` (Characters, Setting, and Scene-by-scene Outline).
3.  **Story (`pipeline/story.py`)**: 
    - **Input**: `output/plan.json`.
    - **Output**: `output/story.json` (The final narrative text).

## How to Run

### Unified Execution
You can run the entire pipeline using the provided `main.py`:
```bash
python main.py
```

### Individual Stages
Each script in the `pipeline/` directory can also be run independently for testing, provided the previous stage's output exists in the `output/` folder.

## Key Components

- **`models/llm_client.py`**: Centralized LLM interface with 4-bit quantization and retry logic.
- **`utils/config.py`**: Shared configuration and prompt building utilities.
- **`notebooks/`**: Interactive demonstration of the generation process.

## Requirements
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```
