import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.inference import HarmonyTTSInference


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="config.yaml")
    p.add_argument("--text",        required=True, help="Text to synthesize")
    p.add_argument("--description", default="A female speaker delivers a cheerful and clear speech.")
    p.add_argument("--model-path",  default=None)
    p.add_argument("--output",      default="output.wav")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    runner = HarmonyTTSInference(cfg, model_path=args.model_path)
    runner.generate(args.text, args.description, output_path=args.output)


if __name__ == "__main__":
    main()
