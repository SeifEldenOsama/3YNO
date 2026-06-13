import modal
import os
import sys
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# Define local project path for Modal
LOCAL_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "git+https://github.com/huggingface/transformers.git",
        "torch",
        "torchvision",
        "torchaudio",
        "pillow",
        "pyyaml",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "fastapi[standard]"
    )
    .add_local_dir(os.path.join(LOCAL_PROJECT_ROOT, "src"), remote_path="/root/src")
    .add_local_file(os.path.join(LOCAL_PROJECT_ROOT, "config.yaml"), remote_path="/root/config.yaml")
)

app = modal.App("glm-ocr-api", image=image)
VOLUME_NAME = "glm-ocr-cache"
CACHE_DIR = "/model-cache"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.cls(
    gpu="A10G",
    volumes={CACHE_DIR: volume},
    timeout=3600,
)
class GLMOCRModel:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, "/root")
        from src.config import Config
        from src.inference import GLMOCRInference
        
        cfg = Config.load("/root/config.yaml")
        os.environ["HF_HOME"] = CACHE_DIR
        self.infer = GLMOCRInference(cfg)
        self.infer.load()
        volume.commit() # Ensure model is persisted to volume

    @modal.method()
    def process(self, image_bytes: bytes, prompt: str = None):
        temp_path = "/tmp/api_image.png"
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        return self.infer.run(temp_path, prompt=prompt)

@app.function(image=image)
@modal.fastapi_endpoint(method="POST")
async def ocr(file: UploadFile = File(...), prompt: str = "Text Recognition:"):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_bytes = await file.read()
    result = GLMOCRModel().process.remote(image_bytes, prompt=prompt)
    
    return JSONResponse(content={"lesson_text": result})
