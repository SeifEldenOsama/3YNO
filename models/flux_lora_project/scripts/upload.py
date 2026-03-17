from __future__ import annotations
import argparse
import sys
import os
import subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.config import load_config
from src.uploader import HubUploader


def parse_args():
    p = argparse.ArgumentParser(description="Upload/download FLUX LoRA via HuggingFace Hub")
    p.add_argument("--config",     default="config.yaml", help="Path to config.yaml")
    p.add_argument("--path",       default=None, help="Local directory to upload from / download to")
    p.add_argument("--repo",       default=None, help="Override HF repo id (e.g. username/my-lora)")
    p.add_argument("--download",   action="store_true", help="Download instead of upload")
    p.add_argument("--private",    action="store_true", help="Make HF repo private")
    p.add_argument("--from-modal", action="store_true", help="Download from Modal volume first, then upload to HF")
    return p.parse_args()


def download_from_modal(volume_name: str, local_path: str):
    subprocess.run([
        "modal", "volume", "get",
        volume_name,
        "flux-lora-output",
        local_path,
    ], check=True)
    print(f"Downloaded to: {local_path}")


def main():
    args     = parse_args()
    cfg      = load_config(args.config)
    if args.repo:    cfg.hub.repo_id = args.repo
    if args.private: cfg.hub.private = True

    local_path = args.path or cfg.checkpointing.local_output

    if args.from_modal:
        download_from_modal(cfg.modal.volume_name, local_path)

    uploader = HubUploader(cfg)

    print("=" * 55)
    print("  HuggingFace Hub — Upload / Download")
    print("=" * 55)
    print(f"  Repo    : {cfg.hub.repo_id}")
    print(f"  Mode    : {'Download' if args.download else 'Upload'}")
    print(f"  Path    : {local_path}")
    print("=" * 55 + "\n")

    if args.download:
        uploader.download(local_path=local_path)
    else:
        uploader.upload(local_path=local_path)


if __name__ == "__main__":
    main()
