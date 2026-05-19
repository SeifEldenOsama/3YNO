import modal
import os
import sys

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
        "protobuf"
    )
    .add_local_dir(os.path.join(LOCAL_PROJECT_ROOT, "src"), remote_path="/root/src")
    .add_local_file(os.path.join(LOCAL_PROJECT_ROOT, "config.yaml"), remote_path="/root/config.yaml")
)

app = modal.App("glm-ocr-cloud", image=image)
# We'll load the config inside the container to ensure it uses the remote path
# But for the volume name, we can use a default or hardcode it if needed
VOLUME_NAME = "glm-ocr-cache"
CACHE_DIR = "/model-cache"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.cls(
    gpu="A10G", # Default GPU
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
        temp_path = "/tmp/input_image.png"
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
            
        return self.infer.run(temp_path, prompt=prompt)

@app.local_entrypoint()
def main(image_path: str, prompt: str = "Text Recognition:"):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    model = GLMOCRModel()
    result = model.process.remote(image_bytes, prompt=prompt)
    print("\n--- Remote OCR Result ---")
    print(result)
