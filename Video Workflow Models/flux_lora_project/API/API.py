import modal
import io
import os
import subprocess
from dotenv import load_dotenv, find_dotenv
from fastapi.responses import Response
from pydantic import BaseModel

load_dotenv(find_dotenv())

if not os.environ.get("MODAL_TASK_ID"):
    HF_TOKEN = os.environ["HF_TOKEN"]
    subprocess.run(
        ["modal", "secret", "create", "my-huggingface-secret", f"HF_TOKEN={HF_TOKEN}", "--force"],
        check=True,
    )

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "fastapi[standard]",
    )
)

app = modal.App("flux-base-api", image=image)

MODEL_ID = "black-forest-labs/FLUX.1-dev"

volume = modal.Volume.from_name("flux-base-cache", create_if_missing=True)
CACHE_DIR = "/model-cache"


@app.cls(
    gpu="A100",
    volumes={CACHE_DIR: volume},
    timeout=3600,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
class FluxModel:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import FluxPipeline

        os.environ["HF_HOME"] = CACHE_DIR
        self.pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=os.environ["HF_TOKEN"],
            cache_dir=CACHE_DIR,
        ).to("cuda")
        volume.commit()
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


class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    seed: int = 42
    width: int = 1024
    height: int = 1024


@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
def generate(req: GenerateRequest) -> Response:
    from fastapi import HTTPException

    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="`prompt` must not be empty.")

    png_bytes = FluxModel().generate.remote(
        req.prompt,
        req.num_inference_steps,
        req.guidance_scale,
        req.seed,
        req.width,
        req.height,
    )
    return Response(content=png_bytes, media_type="image/png")