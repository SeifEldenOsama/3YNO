import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.trainer import HarmonyTTSTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if args.output:
        cfg.training.output_dir  = args.output
        cfg.training.local_output = args.output

    trainer = HarmonyTTSTrainer(
        cfg        = cfg,
        output_dir = cfg.training.local_output,
    )
    trainer.run()


if __name__ == "__main__":
    main()
