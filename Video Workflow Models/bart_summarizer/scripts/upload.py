import argparse, sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.uploader import HubUploader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        default="config.yaml")
    p.add_argument("--path",          default=None)
    p.add_argument("--repo",          default=None)
    p.add_argument("--download",      action="store_true")
    p.add_argument("--from-modal",    action="store_true",
                   help="Download from Modal volume first, then upload to HF")
    return p.parse_args()


def download_from_modal(volume_name: str, local_path: str):
    print(f"Downloading from Modal volume: {volume_name}")
    subprocess.run([
        "modal", "volume", "get",
        volume_name,
        "led-summarizer-output",
        local_path,
    ], check=True)
    print(f"Downloaded to: {local_path}")


def main():
    args     = parse_args()
    cfg      = load_config(args.config)
    if args.repo: cfg.hub.repo_id = args.repo

    local_path = args.path or cfg.output.local_dir

    if args.from_modal:
        download_from_modal(cfg.modal.volume_name, local_path)

    uploader = HubUploader(cfg)

    if args.download:
        uploader.download(local_path=local_path)
    else:
        uploader.upload(local_path=local_path)


if __name__ == "__main__":
    main()
