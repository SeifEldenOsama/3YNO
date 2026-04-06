from __future__ import annotations
import os
import subprocess
from pathlib import Path

from src.config import Config


def _patch_parler_training_script(repo_path: Path):
    import training.data as data_module

    data_py = Path(data_module.__file__)
    lines   = data_py.read_text().splitlines()

    patched = []
    for line in lines:
        if 'vectorized_datasets["validation"]' in line:
            line = line.replace('vectorized_datasets["validation"]', 'vectorized_datasets["eval"]')
        patched.append(line)

    data_py.write_text("\n".join(patched))

    script = repo_path / "training" / "run_parler_tts_training.py"
    lines  = script.read_text().splitlines()

    patched = []
    for line in lines:
        if "num_proc=min(data_args.preprocessing_num_workers" in line:
            line = "                num_proc=1,"
        if '"evaluation_strategy"' in line:
            line = line.replace('"evaluation_strategy"', '"eval_strategy"')
        patched.append(line)

    script.write_text("\n".join(patched))
    print("Parler-TTS training script patched.")


class HarmonyTTSTrainer:
    def __init__(self, cfg: Config, output_dir: str, volume=None):
        self.cfg        = cfg
        self.output_dir = output_dir
        self.volume     = volume

    def run(self):
        import sys
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        hf_token = self.cfg.credentials.hf_token
        if hf_token:
            os.environ["HF_TOKEN"]               = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
            from huggingface_hub import login
            login(token=hf_token)

        repo_path = Path("/root/parler-tts")
        if not repo_path.exists():
            print("Cloning Parler-TTS repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/huggingface/parler-tts.git", str(repo_path)],
                check=True,
            )

        sys.path.insert(0, str(repo_path))
        _patch_parler_training_script(repo_path)

        d        = self.cfg.dataset
        m        = self.cfg.model
        t        = self.cfg.training
        num_gpus = int(self.cfg.modal.gpu.split(":")[1]) if ":" in self.cfg.modal.gpu else 1

        cmd = f"""
accelerate launch \\
  --num_processes={num_gpus} \\
  --mixed_precision=bf16 \\
  --num_machines=1 \\
  --dynamo_backend=no \\
  training/run_parler_tts_training.py \\
  --model_name_or_path "{m.name}" \\
  --train_dataset_name "{d.repo_id}" \\
  --train_metadata_dataset_name "{d.repo_id}" \\
  --train_dataset_config_name "default" \\
  --train_split_name "{d.train_split}" \\
  --eval_dataset_name "{d.repo_id}" \\
  --eval_metadata_dataset_name "{d.repo_id}" \\
  --eval_dataset_config_name "default" \\
  --eval_split_name "{d.eval_split}" \\
  --max_train_samples {d.max_train_samples} \\
  --max_eval_samples {d.max_eval_samples} \\
  --seed {t.seed} \\
  --do_train true \\
  --do_eval true \\
  --preprocessing_num_workers 1 \\
  --eval_strategy "steps" \\
  --eval_steps {t.eval_steps} \\
  --save_steps {t.save_steps} \\
  --description_column_name "{d.description_col}" \\
  --prompt_column_name "{d.prompt_col}" \\
  --target_audio_column_name "{d.audio_col}" \\
  --description_tokenizer_name "{m.description_tokenizer}" \\
  --prompt_tokenizer_name "{m.name}" \\
  --save_to_disk "/tmp/parler_dataset_processed" \\
  --temporary_save_to_disk "/tmp/parler_dataset_temp" \\
  --output_dir "{self.output_dir}" \\
  --overwrite_output_dir true \\
  --per_device_train_batch_size {t.per_device_train_batch} \\
  --per_device_eval_batch_size {t.per_device_eval_batch} \\
  --gradient_accumulation_steps {t.gradient_accum_steps} \\
  --gradient_checkpointing {str(t.gradient_checkpointing).lower()} \\
  --optim "adamw_bnb_8bit" \\
  --learning_rate {t.learning_rate} \\
  --lr_scheduler_type "{t.lr_scheduler}" \\
  --warmup_steps {t.warmup_steps} \\
  --weight_decay {t.weight_decay} \\
  --max_steps {t.max_steps} \\
  --bf16 {str(t.bf16).lower()} \\
  --report_to "none"
"""

        print("Starting Harmony TTS full fine-tuning...")
        print(f"  Model   : {m.name}")
        print(f"  Dataset : {d.repo_id}")
        print(f"  Train   : {d.max_train_samples} samples")
        print(f"  Eval    : {d.max_eval_samples} samples")
        print(f"  Steps   : {t.max_steps}")
        print(f"  Output  : {self.output_dir}")

        subprocess.run(cmd, shell=True, check=True, cwd=str(repo_path))

        if self.volume:
            self.volume.commit()

        print(f"Training complete. Model saved to: {self.output_dir}")
