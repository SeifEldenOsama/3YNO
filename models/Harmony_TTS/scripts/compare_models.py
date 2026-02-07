import modal
import os
from pathlib import Path

GPU_CONFIG = "H100:1"
VOLUME_NAME = "tts-dataset-storage"
MOUNT_PATH = Path("/data")
FINETUNED_MODEL_PATH = MOUNT_PATH / "parler-tts-finetuned-h100"
BASE_MODEL_NAME = "parler-tts/parler-tts-mini-v1"

REQUIREMENTS = [
    "torch==2.4.1",
    "torchaudio==2.4.1",
    "transformers==4.46.1",
    "parler-tts @ git+https://github.com/huggingface/parler-tts.git",
    "soundfile",
    "scipy",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        *REQUIREMENTS,
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

app = modal.App("parler-tts-comparison", image=image)

@app.function(
    volumes={str(MOUNT_PATH): modal.Volume.from_name(VOLUME_NAME)},
    gpu=GPU_CONFIG,
    timeout=600,
    env={
        "FORCE_LIBSNDFILE": "1",
        "HF_AUDIO_DISABLE_TORCHCODEC": "1",
    },
)
def run_comparison(prompt: str, description: str):
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    import soundfile as sf

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_audio(model_id, label):
        print(f"Loading {label} model…")
        model = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
        model.eval()

        prompt_tokenizer = AutoTokenizer.from_pretrained(model_id)
        description_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

        with torch.inference_mode():
            input_ids = description_tokenizer(
                description, return_tensors="pt"
            ).input_ids.to(device)

            prompt_input_ids = prompt_tokenizer(
                prompt, return_tensors="pt"
            ).input_ids.to(device)

            audio = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
            )

        audio_arr = audio.cpu().numpy().squeeze()
        filename = f"output_{label.lower().replace(' ', '_')}.wav"
        sf.write(filename, audio_arr, model.config.sampling_rate)

        with open(filename, "rb") as f:
            return f.read(), filename

    base_audio, base_file = generate_audio(BASE_MODEL_NAME, "Base")

    if not FINETUNED_MODEL_PATH.exists():
        return f"Fine-tuned model not found at {FINETUNED_MODEL_PATH}", None

    ft_audio, ft_file = generate_audio(str(FINETUNED_MODEL_PATH), "Fine-tuned")

    return {
        "base": (base_audio, base_file),
        "finetuned": (ft_audio, ft_file),
    }

@app.local_entrypoint()
def main():
    test_prompt = (
        "Well, when you play super hard, your muscles get tiny little changes that make them feel a bit tired. But don't you worry, those changes are actually helping them get stronger so you can play even more!"
    )
    test_description = (
        "A male speaker delivers a gentle and moderate-paced speech. The recording is clean with a natural quality. The voice has a neutral pitch."
    )

    print("Starting comparison inference…")
    results = run_comparison.remote(test_prompt, test_description)

    if isinstance(results, str):
        print(results)
        return

    for key, (audio_data, filename) in results.items():
        with open(filename, "wb") as f:
            f.write(audio_data)
        print(f"Saved {key} result to {filename}")

    print("\nComparison complete! Listen to both files.")
