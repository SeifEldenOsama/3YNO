import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.trainer import LEDSummarizerTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--csv",    default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if args.epochs: cfg.training.epochs  = args.epochs
    if args.output: cfg.output.local_dir = args.output
    csv = args.csv or cfg.dataset.csv_path

    trainer = LEDSummarizerTrainer(
        cfg        = cfg,
        csv_path   = csv,
        output_dir = cfg.output.local_dir,
    )
    trainer.run()


if __name__ == "__main__":
    main()
