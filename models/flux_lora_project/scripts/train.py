from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.dataset import build_dataset
from src.trainer import FluxLoraTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train FLUX LoRA — local run")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--steps",  type=int,   default=None, help="Override max_steps")
    p.add_argument("--lr",     type=float, default=None, help="Override learning_rate")
    p.add_argument("--rank",   type=int,   default=None, help="Override LoRA rank")
    p.add_argument("--output", default=None, help="Override output directory")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if args.steps:  cfg.training.max_steps      = args.steps
    if args.lr:     cfg.training.learning_rate   = args.lr
    if args.rank:   cfg.lora.rank                = args.rank
    if args.output: cfg.checkpointing.local_output = args.output

    output_dir = cfg.checkpointing.local_output

    print("=" * 55)
    print("  FLUX LoRA Training — Local Run")
    print("=" * 55)
    print(f"  Config   : {args.config}")
    print(f"  Model    : {cfg.model.name}")
    print(f"  Dataset  : {cfg.dataset.source} / {cfg.dataset.kaggle_dataset or cfg.dataset.hf_dataset}")
    print(f"  Steps    : {cfg.training.max_steps}")
    print(f"  Output   : {output_dir}")
    print("=" * 55 + "\n")

    dataset = build_dataset(cfg)
    trainer = FluxLoraTrainer(cfg=cfg, dataset=dataset, output_dir=output_dir)
    trainer.run()


if __name__ == "__main__":
    main()
