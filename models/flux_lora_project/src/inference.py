from __future__ import annotations

import os
from pathlib import Path

import torch

from src.config import Config


class FluxLoraInference:
    def __init__(self, cfg: Config, lora_path: str | None = None):
        self.cfg      = cfg
        self.lora_path = lora_path or self._find_latest_checkpoint()

    def _find_latest_checkpoint(self) -> str:
        import glob
        output_dir  = self.cfg.checkpointing.output_dir
        checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))
        if checkpoints:
            path = checkpoints[-1]
            print(f"Auto-detected latest checkpoint: {path}")
            return path
        print(f"No checkpoint found, using output dir: {output_dir}")
        return output_dir


    def _load_pipeline(self):
        from diffusers import FluxPipeline
        from huggingface_hub import login
        from peft import PeftModel

        login(token=self.cfg.credentials.hf_token)

        print(f"Loading base model: {self.cfg.model.name} ...")
        pipe = FluxPipeline.from_pretrained(
            self.cfg.model.name,
            torch_dtype = torch.bfloat16,
            token       = self.cfg.credentials.hf_token,
        ).to("cuda")

        print(f"Loading LoRA weights from: {self.lora_path} ...")
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer,
            self.lora_path,
            adapter_name="default",
        )
        pipe.transformer = pipe.transformer.merge_and_unload(safe_merge=True)
        print("LoRA merged into transformer")

        return pipe


    def _print_lora_files(self):
        if os.path.isdir(self.lora_path):
            files = os.listdir(self.lora_path)
            print(f"Files in LoRA dir ({self.lora_path}):")
            for f in files:
                print(f"   → {f}")


    def generate(
        self,
        prompt:              str   | None = None,
        num_images:          int   | None = None,
        num_inference_steps: int   | None = None,
        guidance_scale:      float | None = None,
        seed:                int   | None = None,
        output_dir:          str   | None = None,
    ) -> list:
        """Generate images and save them. Returns list of saved file paths."""
        ic = self.cfg.inference

        prompt              = prompt              or ic.prompt
        num_images          = num_images          or ic.num_images
        num_inference_steps = num_inference_steps or ic.num_inference_steps
        guidance_scale      = guidance_scale      or ic.guidance_scale
        seed                = seed                if seed is not None else ic.seed
        output_dir          = output_dir          or ic.local_output

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._print_lora_files()

        pipe = self._load_pipeline()

        print(f"\nGenerating {num_images} image(s) ...")
        print(f"   Prompt : {prompt}")
        print(f"   Steps  : {num_inference_steps}")
        print(f"   CFG    : {guidance_scale}")
        print(f"   Seed   : {seed}\n")

        images = pipe(
            prompt               = prompt,
            num_images_per_prompt = num_images,
            num_inference_steps  = num_inference_steps,
            guidance_scale       = guidance_scale,
            generator            = torch.Generator("cuda").manual_seed(seed),
        ).images

        saved = []
        for i, img in enumerate(images):
            path = os.path.join(output_dir, f"result_{i:02d}.png")
            img.save(path)
            print(f"Saved → {path}")
            saved.append(path)

        print(f"\nDone! {len(saved)} image(s) saved to: {output_dir}")
        return saved
