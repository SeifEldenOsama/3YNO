# 3YNO: Content Extraction & Summarization

This module serves as the foundational layer of the 3YNO pipeline. Its primary role is to ingest dense educational materials (books, research papers, etc.) and distill them into core scientific concepts that can be later transformed into narratives.

## Role in 3YNO

For dyslexic and visual learners, large blocks of text can be overwhelming. This module uses the **Longformer Encoder-Decoder (LED)** architecture to:
- Process long-form educational content (up to 16,384 tokens).
- Extract essential scientific facts and key takeaways.
- Provide a structured summary that serves as the "script" for the narrative transformation phase.

## Technical Overview

- **Model**: `allenai/led-base-16384`
- **Focus**: High-fidelity extraction of educational content from long documents.

## Usage

Ensure your educational source files are in the `data/` directory, then run:
```bash
python src/generate.py --text "Path to educational text"
```
