from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.inference import FluxLoraInference


def parse_args():
    p = argparse.ArgumentParser(description="Generate images with FLUX LoRA — local run")
    p.add_argument("--config",     default="config.yaml", help="Path to config.yaml")
    p.add_argument("--prompt",     default=None, help="Override inference prompt")
    p.add_argument("--num-images", type=int,   default=None, help="Number of images")
    p.add_argument("--steps",      type=int,   default=None, help="Inference steps")
    p.add_argument("--cfg-scale",  type=float, default=None, help="Guidance scale")
    p.add_argument("--lora-scale", type=float, default=None, help="LoRA scale")
    p.add_argument("--seed",       type=int,   default=None, help="Random seed")
    p.add_argument("--lora-path",  default=None, help="Override LoRA weights path")
    p.add_argument("--output",     default=None, help="Override output directory")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if args.prompt:     cfg.inference.prompt              = args.prompt
    if args.num_images: cfg.inference.num_images          = args.num_images
    if args.steps:      cfg.inference.num_inference_steps = args.steps
    if args.cfg_scale:  cfg.inference.guidance_scale      = args.cfg_scale
    if args.lora_scale: cfg.inference.lora_scale          = args.lora_scale
    if args.seed is not None: cfg.inference.seed          = args.seed
    if args.output:     cfg.inference.local_output        = args.output

    print("=" * 55)
    print("  FLUX LoRA Inference — Local Run")
    print("=" * 55)
    print(f"  Prompt  : {cfg.inference.prompt}")
    print(f"  Images  : {cfg.inference.num_images}")
    print(f"  Steps   : {cfg.inference.num_inference_steps}")
    print(f"  LoRA    : {args.lora_path or 'auto-detect'}")
    print(f"  Output  : {cfg.inference.local_output}")
    print("=" * 55 + "\n")

    runner = FluxLoraInference(cfg, lora_path=args.lora_path)
    runner.generate(output_dir=cfg.inference.local_output)


if __name__ == "__main__":
    main()
