import modal
import os
import json
import base64
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

load_dotenv(find_dotenv())

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "google-genai",
        "fastapi[standard]",
        "python-dotenv",
    )
    .add_local_dir("src", remote_path="/root/project/src")
)

app = modal.App("gemini-tts-api", image=image)

GEMINI_API_KEYS = os.environ.get("GEMINI_API_KEYS", "")


@app.function(
    image=image,
    secrets=[modal.Secret.from_dict({"GEMINI_API_KEYS": GEMINI_API_KEYS})],
    timeout=500,
    memory=512,
)
def generate_tts(description: str, text: str) -> bytes:
    """Generate single TTS audio using Gemini with random key fallback."""
    import sys
    sys.path.insert(0, "/root/project")
    from src.inference import GeminiTTSInference
    tts = GeminiTTSInference()
    return tts.generate_bytes(text=text, description=description)


@app.function(
    image=image,
    secrets=[modal.Secret.from_dict({"GEMINI_API_KEYS": GEMINI_API_KEYS})],
    timeout=500,
    memory=1024,
)
def generate_tts_batch(requests: list[dict]) -> list[bytes]:
    """Generate multiple TTS clips in parallel, each with a different random key."""
    import sys
    sys.path.insert(0, "/root/project")
    from src.inference import generate_parallel
    return generate_parallel(requests)


web_app = FastAPI(title="Gemini TTS API")


class TTSRequest(BaseModel):
    description: str = "Aoede A calm and friendly female voice with a warm clear tone."
    text: str


class TTSBatchItem(BaseModel):
    description: str
    text: str


class TTSBatchRequest(BaseModel):
    requests: list[TTSBatchItem]


@app.function(
    image=image,
    memory=512,
    timeout=500,
)
@modal.asgi_app()
def fastapi_app():

    @web_app.post("/synthesize", response_class=Response)
    async def synthesize(req: TTSRequest):
        """Generate a single voice clip."""
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="`text` must not be empty.")
        wav_bytes = await generate_tts.remote.aio(req.description, req.text)
        return Response(content=wav_bytes, media_type="audio/wav")

    @web_app.post("/synthesize-batch")
    async def synthesize_batch(req: TTSBatchRequest):
        """Generate multiple voice clips in parallel, each using a different API key.
        Returns a JSON list of base64-encoded WAV files in the same order as input.
        """
        if not req.requests:
            raise HTTPException(status_code=400, detail="requests list is empty.")
        if len(req.requests) > 50:
            raise HTTPException(status_code=400, detail="Max 50 requests per batch.")

        raw_requests = [{"text": r.text, "description": r.description} for r in req.requests]
        wav_list = await generate_tts_batch.remote.aio(raw_requests)

        return JSONResponse(content={
            "count": len(wav_list),
            "results": [
                {
                    "index": i,
                    "audio_base64": base64.b64encode(wav).decode("utf-8"),
                    "format": "wav",
                }
                for i, wav in enumerate(wav_list)
            ]
        })

    @web_app.get("/health")
    async def health():
        return {"status": "ok"}

    return web_app