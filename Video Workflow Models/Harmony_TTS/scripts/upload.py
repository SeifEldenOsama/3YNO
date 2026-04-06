import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config
from src.uploader import HubUploader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="config.yaml")
    p.add_argument("--path",     default=None)
    p.add_argument("--repo",     default=None)
    p.add_argument("--download", action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    cfg      = load_config(args.config)
    if args.repo: cfg.hub.repo_id = args.repo
    uploader = HubUploader(cfg)

    if args.download:
        uploader.download(local_path=args.path)
    else:
        uploader.upload(local_path=args.path)


if __name__ == "__main__":
    main()
