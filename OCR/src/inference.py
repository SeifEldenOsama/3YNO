import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from src.config import Config

class GLMOCRInference:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.processor = None
        self.model = None

    def load(self):
        print(f"Loading model: {self.cfg.model.name}...")
        self.processor = AutoProcessor.from_pretrained(self.cfg.model.name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.model.name,
            torch_dtype=torch.bfloat16 if self.cfg.model.dtype == "auto" else getattr(torch, self.cfg.model.dtype),
            device_map=self.cfg.model.device,
            trust_remote_code=True
        )
        print("Model loaded successfully.")

    def run(self, image_path: str, prompt: str = None, max_new_tokens: int = None):
        if self.model is None:
            self.load()

        prompt = prompt or self.cfg.inference.prompt
        max_new_tokens = max_new_tokens or self.cfg.inference.max_new_tokens

        image = Image.open(image_path).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        inputs.pop("token_type_ids", None)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output_text = self.processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return output_text
