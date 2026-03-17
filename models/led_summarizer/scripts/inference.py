import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.inference import LEDSummarizerInference


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="config.yaml")
    p.add_argument("--text",       default=None)
    p.add_argument("--csv",        default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--output",     default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    runner = LEDSummarizerInference(cfg, model_path=args.model_path)

    if args.csv:
        runner.summarize_csv(args.csv, output_path=args.output)
    elif args.text:
        print(runner.summarize(args.text))
    else:
        print("Provide --text or --csv")


if __name__ == "__main__":
    main()
