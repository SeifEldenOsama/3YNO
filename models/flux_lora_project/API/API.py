import modal
import io

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "diffusers",
        "transformers",
        "peft",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "fastapi[standard]",
    )
)

app = modal.App("flux-lora-api", image=image)

MODEL_ID  = "black-forest-labs/FLUX.1-dev"
LORA_ID   = "SeifElden2342532/flux-lora-characters"
HF_TOKEN  = "YOUR TOKEN HERE"

volume = modal.Volume.from_name("flux-lora-cache", create_if_missing=True)
CACHE_DIR = "/model-cache"


@app.cls(
    gpu="A100",
    volumes={CACHE_DIR: volume},
    timeout=600,
    scaledown_window=120,
)
class FluxModel:

    @modal.enter()
    def load(self):
        import torch
        from diffusers import FluxPipeline
        from peft import PeftModel

        self.pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN,
            cache_dir=CACHE_DIR,
        ).to("cuda")

        self.pipe.transformer = PeftModel.from_pretrained(
            self.pipe.transformer,
            LORA_ID,
            subfolder="flux-lora-output",
            token=HF_TOKEN,
            cache_dir=CACHE_DIR,
        )
        self.pipe.transformer = self.pipe.transformer.merge_and_unload()
        print("Model ready ✓")

    @modal.method()
    def generate(
        self,
        prompt: str,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int = 42,
        width: int = 1024,
        height: int = 1024,
    ) -> bytes:
        import torch

        generator = torch.Generator("cuda").manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()


from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

web_app = FastAPI(title="FLUX LoRA Image API")


class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    seed: int = 42
    width: int = 1024
    height: int = 1024


@app.function()
@modal.asgi_app()
def fastapi_app():

    @web_app.post("/generate", response_class=Response)
    async def generate(req: GenerateRequest):
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="`prompt` must not be empty.")
        flux = FluxModel()
        png_bytes = flux.generate.remote(
            req.prompt,
            req.num_inference_steps,
            req.guidance_scale,
            req.seed,
            req.width,
            req.height,
        )
        return Response(content=png_bytes, media_type="image/png")

    @web_app.get("/health")
    async def health():
        return {"status": "ok"}

    return web_app