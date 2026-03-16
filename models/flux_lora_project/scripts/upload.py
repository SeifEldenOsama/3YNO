from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.uploader import HubUploader


def parse_args():
    p = argparse.ArgumentParser(description="Upload/download FLUX LoRA via HuggingFace Hub")
    p.add_argument("--config",   default="config.yaml", help="Path to config.yaml")
    p.add_argument("--path",     default=None, help="Local directory to upload from / download to")
    p.add_argument("--repo",     default=None, help="Override HF repo id (e.g. username/my-lora)")
    p.add_argument("--download", action="store_true", help="Download instead of upload")
    p.add_argument("--private",  action="store_true", help="Make HF repo private")
    return p.parse_args()


def main():
    args     = parse_args()
    cfg      = load_config(args.config)
    uploader = HubUploader(cfg)

    if args.repo:    cfg.hub.repo_id  = args.repo
    if args.private: cfg.hub.private  = True

    print("=" * 55)
    print("  HuggingFace Hub — Upload / Download")
    print("=" * 55)
    print(f"  Repo    : {cfg.hub.repo_id}")
    print(f"  Mode    : {'Download' if args.download else 'Upload'}")
    print(f"  Path    : {args.path or '(from config)'}")
    print("=" * 55 + "\n")

    if args.download:
        uploader.download(local_path=args.path)
    else:
        uploader.upload(local_path=args.path)


if __name__ == "__main__":
    main()
