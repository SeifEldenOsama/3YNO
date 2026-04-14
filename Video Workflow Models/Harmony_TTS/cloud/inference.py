import modal
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

PYTHON_VERSION  = "3.11"
GEMINI_API_KEYS = os.environ.get("GEMINI_API_KEYS", "")

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        "google-genai",
        "python-dotenv",
    )
    .add_local_dir("src", remote_path="/root/project/src")
)

app = modal.App("gemini-tts", image=image)


@app.function(
    image=image,
    secrets=[modal.Secret.from_dict({"GEMINI_API_KEYS": GEMINI_API_KEYS})],
    timeout=120,
    memory=512,
)
def generate_remote(
    text:        str = "Hello, this is Gemini TTS speaking.",
    description: str = "Aoede A calm and friendly female voice with a warm clear tone.",
) -> bytes:
    import sys
    sys.path.insert(0, "/root/project")
    from src.inference import GeminiTTSInference

    tts = GeminiTTSInference()
    return tts.generate_bytes(text=text, description=description)


@app.local_entrypoint()
def main(
    text:        str = "Hello, this is Gemini TTS speaking.",
    description: str = "Aoede A calm and friendly female voice with a warm clear tone.",
    output:      str = "output.wav",
):
    audio_bytes = generate_remote.remote(text=text, description=description)
    with open(output, "wb") as f:
        f.write(audio_bytes)
    print(f"Audio saved to: {output}")