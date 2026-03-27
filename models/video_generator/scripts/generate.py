import argparse
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.generator import VideoGenerator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="config.yaml")
    p.add_argument("--image",   required=True, help="Path to frame PNG")
    p.add_argument("--audio",   required=True, help="Path to audio WAV")
    p.add_argument("--prompt",  default=None,  help="LTX-2 video prompt (optional)")
    p.add_argument("--output",  default="output_hq.mp4")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    gen  = VideoGenerator(cfg)
    gen.load_model()

    image_bytes = Path(args.image).read_bytes()
    audio_bytes = Path(args.audio).read_bytes()

    video_bytes = gen.generate(
        image_bytes=image_bytes,
        audio_bytes=audio_bytes,
        prompt=args.prompt,
        seed=args.seed,
    )

    Path(args.output).write_bytes(video_bytes)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
