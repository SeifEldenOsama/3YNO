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
    p.add_argument("--frame",   required=True, help="Path to frame.png")
    p.add_argument("--audio",   required=True, help="Path to audio.wav")
    p.add_argument("--prompt",  required=True, help="LTX-2 video prompt")
    p.add_argument("--output",  default="output.mp4")
    p.add_argument("--seed",    type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    gen  = VideoGenerator(cfg)
    gen.load_model()

    frame_bytes = Path(args.frame).read_bytes()
    audio_bytes = Path(args.audio).read_bytes()

    clip_bytes, last_frame_bytes = gen.generate_shot(
        frame_bytes = frame_bytes,
        audio_bytes = audio_bytes,
        prompt      = args.prompt,
        seed        = args.seed,
    )

    Path(args.output).write_bytes(clip_bytes)
    last_frame_path = args.output.replace(".mp4", "_last_frame.png")
    Path(last_frame_path).write_bytes(last_frame_bytes)

    print(f"Clip saved       : {args.output}")
    print(f"Last frame saved : {last_frame_path}")


if __name__ == "__main__":
    main()
