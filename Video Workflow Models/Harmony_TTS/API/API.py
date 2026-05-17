import modal
import os
from dotenv import load_dotenv, find_dotenv
from fastapi.responses import Response
from pydantic import BaseModel

load_dotenv(find_dotenv())

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "requests",
        "fastapi[standard]",
        "python-dotenv",
    )
    .add_local_dir("src", remote_path="/root/project/src")
)

app = modal.App("gemini-tts-api", image=image)

GEMINI_API_KEYS = os.environ.get("GEMINI_API_KEYS", "")


class TTSRequest(BaseModel):
    description: str = "Aoede A calm and friendly female voice with a warm clear tone."
    text: str


@app.function(
    image=image,
    secrets=[modal.Secret.from_dict({"GEMINI_API_KEYS": GEMINI_API_KEYS})],
    timeout=500,
    memory=512,
)
@modal.fastapi_endpoint(method="POST")
def synthesize(req: TTSRequest) -> Response:
    import sys
    sys.path.insert(0, "/root/project")
    from src.inference import GeminiTTSInference
    if not req.text.strip():
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="`text` must not be empty.")
    tts = GeminiTTSInference()
    wav_bytes = tts.generate_bytes(text=req.text, description=req.description)
    return Response(content=wav_bytes, media_type="audio/wav")