from __future__ import annotations

import os
from pathlib import Path

from src.config import Config


class HubUploader:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def upload(self, local_path: str | None = None):
        """Upload LoRA weights from local_path (or config output_dir) to HF Hub."""
        from huggingface_hub import HfApi, login, create_repo

        login(token=self.cfg.credentials.hf_token)
        api     = HfApi()
        repo_id = self.cfg.hub.repo_id

        if not repo_id:
            raise ValueError("hub.repo_id is empty in config.yaml")

        # Create repo if it doesn't exist
        print(f"Creating/verifying repo: {repo_id} ...")
        create_repo(
            repo_id  = repo_id,
            repo_type = "model",
            private  = self.cfg.hub.private,
            exist_ok = True,
            token    = self.cfg.credentials.hf_token,
        )

        src = local_path or self.cfg.checkpointing.local_output
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Upload source directory not found: {src}")

        print(f"Uploading from: {src}")
        print(f"   → Repo: {repo_id}")

        api.upload_folder(
            repo_id        = repo_id,
            folder_path    = src,
            repo_type      = "model",
            commit_message = self.cfg.hub.commit_message,
            token          = self.cfg.credentials.hf_token,
        )

        url = f"https://huggingface.co/{repo_id}"
        print(f"\nUpload complete!")
        print(f"   {url}")
        return url

    def download(self, local_path: str | None = None) -> str:
        """Download LoRA weights from HF Hub to local_path."""
        from huggingface_hub import snapshot_download, login

        login(token=self.cfg.credentials.hf_token)
        dest = local_path or self.cfg.checkpointing.local_output

        print(f"Downloading: {self.cfg.hub.repo_id} → {dest}")
        path = snapshot_download(
            repo_id   = self.cfg.hub.repo_id,
            local_dir = dest,
            token     = self.cfg.credentials.hf_token,
        )
        print(f"Downloaded to: {path}")
        return path
