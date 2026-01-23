# 3YNO: Narrative Transformation & Character Planning

This module is responsible for the creative "translation" of scientific content into a format that is engaging and accessible for visual learners and dyslexic individuals.

## Role in 3YNO

The Story Generator takes the distilled facts from the Summarizer and:
1.  **Narrative Framing**: Converts abstract scientific concepts into a story-driven format.
2.  **Character Development**: Defines characters that will act as "guides" or "protagonists" in the final educational video.
3.  **Visual Planning**: Outlines the scenes and visual cues that will be used in the video synthesis stage.

## Technical Overview

- **Model**: `Mistral-7B-Instruct-v0.3` (Quantized for efficiency)
- **Pipeline**:
    - `premise.py`: Establishes the educational story's core idea.
    - `plan.py`: Structures the narrative into logical segments.
    - `story.py`: Generates the final script and character interactions.

## Development Status

Currently focusing on improving character consistency and ensuring that the narrative transformation does not lose the accuracy of the original scientific content.
