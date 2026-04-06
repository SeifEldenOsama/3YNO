from __future__ import annotations
import os
from src.config import Config


class HubUploader:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def upload(self, local_path: str | None = None):
        from huggingface_hub import HfApi, login, create_repo

        login(token=self.cfg.credentials.hf_token)
        api     = HfApi()
        repo_id = self.cfg.hub.repo_id

        if not repo_id:
            raise ValueError("hub.repo_id is empty in config.yaml")

        src = local_path or self.cfg.training.local_output
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Upload source not found: {src}")

        create_repo(
            repo_id   = repo_id,
            repo_type = "model",
            private   = self.cfg.hub.private,
            exist_ok  = True,
            token     = self.cfg.credentials.hf_token,
        )

        api.upload_folder(
            repo_id        = repo_id,
            folder_path    = src,
            repo_type      = "model",
            commit_message = self.cfg.hub.commit_message,
            token          = self.cfg.credentials.hf_token,
        )

        print(f"Upload complete: https://huggingface.co/{repo_id}")

    def download(self, local_path: str | None = None) -> str:
        from huggingface_hub import snapshot_download, login

        login(token=self.cfg.credentials.hf_token)
        dest = local_path or self.cfg.training.local_output

        path = snapshot_download(
            repo_id   = self.cfg.hub.repo_id,
            local_dir = dest,
            token     = self.cfg.credentials.hf_token,
        )
        print(f"Downloaded to: {path}")
        return path
