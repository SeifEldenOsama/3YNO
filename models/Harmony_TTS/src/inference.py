from __future__ import annotations
import os
from pathlib import Path

from src.config import Config


class HarmonyTTSInference:
    def __init__(self, cfg: Config, model_path: str | None = None):
        self.cfg        = cfg
        self.model_path = model_path or cfg.training.local_output
        self.model      = None
        self.prompt_tokenizer      = None
        self.description_tokenizer = None
        self.device     = None

    def load(self):
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from: {self.model_path}")

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.model_path
        ).to(self.device)
        self.model.eval()

        self.prompt_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.description_tokenizer
        )
        print("Model loaded.")

    def generate(
        self,
        text: str,
        description: str,
        output_path: str = "output.wav",
    ) -> str:
        import torch
        import soundfile as sf

        if self.model is None:
            self.load()

        input_ids = self.description_tokenizer(
            description, return_tensors="pt"
        ).input_ids.to(self.device)

        prompt_input_ids = self.prompt_tokenizer(
            text, return_tensors="pt"
        ).input_ids.to(self.device)

        with torch.inference_mode():
            audio = self.model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
            )

        audio_arr = audio.cpu().numpy().squeeze()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio_arr, self.model.config.sampling_rate)
        print(f"Audio saved to: {output_path}")
        return output_path
