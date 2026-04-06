import modal
import io

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "parler-tts",
        "transformers",
        "soundfile",
        "numpy",
        "accelerate",
        "fastapi[standard]",
    )
)

app = modal.App("parler-tts-api", image=image)

MODEL_ID = "SeifElden2342532/Harmony_Parler_TTS"

volume = modal.Volume.from_name("parler-tts-cache", create_if_missing=True)
CACHE_DIR = "/model-cache"


@app.cls(
    gpu="T4",
    volumes={CACHE_DIR: volume},
    timeout=300,
    container_idle_timeout=120,
)
class TTSModel:

    @modal.enter()
    def load(self):
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            MODEL_ID, cache_dir=CACHE_DIR
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    @modal.method()
    def generate(self, description: str, prompt_text: str) -> bytes:
        import torch
        import soundfile as sf

        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
            )

        audio = generated.cpu().numpy().squeeze()
        buf = io.BytesIO()
        sf.write(buf, audio, self.model.config.sampling_rate, format="WAV")
        buf.seek(0)
        return buf.read()


from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

web_app = FastAPI(title="Parler TTS API")


class TTSRequest(BaseModel):
    description: str = "A calm male voice with medium speed and clear audio"
    text: str


@app.function()
@modal.asgi_app()
def fastapi_app():

    @web_app.post("/synthesize", response_class=Response)
    async def synthesize(req: TTSRequest):
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="`text` must not be empty.")
        tts = TTSModel()
        wav_bytes = tts.generate.remote(req.description, req.text)
        return Response(content=wav_bytes, media_type="audio/wav")

    @web_app.get("/health")
    async def health():
        return {"status": "ok"}

    return web_app