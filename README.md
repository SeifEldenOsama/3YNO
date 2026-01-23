# 3YNO: Advanced NLP Suite

3YNO is a professional-grade Natural Language Processing (NLP) repository focused on high-quality text generation and analysis. It currently features two primary modules: a sophisticated **Summarizer** and a creative **Story Generator**.

## Project Overview

The 3YNO project aims to provide robust, scalable, and easy-to-use tools for common NLP tasks. By leveraging state-of-the-art transformer models and efficient pipelines, 3YNO enables developers and researchers to integrate advanced language capabilities into their applications.

| Module | Description | Key Technology |
| :--- | :--- | :--- |
| **Summarizer** | Efficiently condenses long-form text into concise summaries. | `allenai/led-base-16384` |
| **Story Generator** | Generates creative and coherent stories from simple premises. | `Mistral-7B-Instruct-v0.3` |

## Repository Structure

The repository is organized into modular components for better maintainability and clarity:

```text
3YNO/
├── models/
│   ├── summarizer/       # Text summarization module
│   └── story_generator/  # Creative story generation module
├── .gitignore            # Standard Python git ignore rules
├── LICENSE               # MIT License
└── README.md             # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-enabled GPU (recommended for model inference)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SeifEldenOsama/3YNO.git
   cd 3YNO
   ```

2. Install the required dependencies for the specific module you wish to use. For example, for the Story Generator:
   ```bash
   pip install -r models/story_generator/requirements.txt
   ```

## Modules

### Summarizer
Located in `models/summarizer/`, this module uses the Longformer Encoder-Decoder (LED) model, which is specifically designed for long documents. It includes scripts for training, evaluation, and inference.

### Story Generator
Located in `models/story_generator/`, this module utilizes the Mistral-7B model with 4-bit quantization to generate creative narratives. It features a structured pipeline for planning and executing story generation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any improvements or bug fixes.
