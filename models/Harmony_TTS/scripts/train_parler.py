%%writefile train_parler.py
import modal
import os
import subprocess
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
GPU_CONFIG = "H100:1"
NUM_GPUS = 1

VOLUME_NAME = "tts-dataset-storage"
MOUNT_PATH = Path("/data")
OUTPUT_DIR = MOUNT_PATH / "parler-tts-finetuned-h100"
HF_DATASET_REPO = "SeifElden2342532/parler-tts-dataset-format"

# -------------------------
# DEPENDENCIES (H100 SAFE)
# -------------------------
REQUIREMENTS = [
    "torch==2.4.1",
    "torchaudio==2.4.1",
    "accelerate",
    "datasets[audio]",
    "transformers==4.46.1",
    "pydantic==1.10.17",
    "tqdm",
    "soundfile",
    "scipy",
    "pyyaml",
    "protobuf==4.25.8",
    "wandb",
    "evaluate",
    "jiwer",
    "librosa",
    "bitsandbytes",
    "huggingface_hub",
    "parler-tts @ git+https://github.com/huggingface/parler-tts.git",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ffmpeg", "libsndfile1")
    .run_commands("ulimit -n 65536")
    .pip_install(
        *REQUIREMENTS,
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

app = modal.App(
    "parler-tts-h100-finetune",
    image=image,
)

# -------------------------
# TRAIN FUNCTION
# -------------------------
@app.function(
    volumes={str(MOUNT_PATH): modal.Volume.from_name(VOLUME_NAME)},
    timeout=25000,
    gpu=GPU_CONFIG,
    env={
        "FORCE_LIBSNDFILE": "1",
        "HF_AUDIO_DISABLE_TORCHCODEC": "1",
    },
)
def finetune_parler_tts():
    repo_path = Path("/root/parler-tts")

    if not repo_path.exists():
        print("Cloning Parler-TTS repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/huggingface/parler-tts.git", str(repo_path)],
            check=True,
        )

    # -------------------------
    # PATCH KNOWN PARLER-TTS BUGS
    # -------------------------
    import training.data

    data_py_path = Path(training.data.__file__)
    content = data_py_path.read_text()

    buggy_code = (
        'metadata_dataset_names = metadata_dataset_names.split("+") '
        'if metadata_dataset_names is not None else None'
    )
    fixed_code = (
        'metadata_dataset_names = metadata_dataset_names.split("+") '
        'if (metadata_dataset_names is not None and isinstance(metadata_dataset_names, str)) '
        'else [None] * len(dataset_names)'
    )
    if buggy_code in content:
        content = content.replace(buggy_code, fixed_code)

    buggy_eval_code = 'vectorized_datasets["validation"]'
    fixed_eval_code = 'vectorized_datasets["eval"]'
    if buggy_eval_code in content:
        content = content.replace(buggy_eval_code, fixed_eval_code)

    data_py_path.write_text(content)

    training_script_path = repo_path / "training" / "run_parler_tts_training.py"
    script_content = training_script_path.read_text()

    buggy_num_proc = (
        'num_proc=min(data_args.preprocessing_num_workers, '
        'len(vectorized_datasets["eval"]) - 1),'
    )
    fixed_num_proc = "num_proc=1,"
    if buggy_num_proc in script_content:
        script_content = script_content.replace(buggy_num_proc, fixed_num_proc)

    # -------------------------
    # IMPLEMENT EARLY STOPPING
    # -------------------------
    # We inject early stopping logic into the training loop
    if "early_stopping_patience = 3" not in script_content:
        # Add early stopping variables
        script_content = script_content.replace(
            "cur_step = 0",
            "cur_step = 0\n    early_stopping_patience = 3\n    best_eval_loss = float('inf')\n    patience_counter = 0"
        )
        
        # Add early stopping check after evaluation
        early_stopping_logic = """
                # Early Stopping Logic
                current_eval_loss = eval_metrics['loss']
                if current_eval_loss < best_eval_loss:
                    best_eval_loss = current_eval_loss
                    patience_counter = 0
                    if accelerator.is_main_process:
                        steps_trained_progress_bar.write(f"New best eval loss: {best_eval_loss}. Saving best model.")
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(os.path.join(training_args.output_dir, 'best_model'))
                else:
                    patience_counter += 1
                    if accelerator.is_main_process:
                        steps_trained_progress_bar.write(f"Eval loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    if accelerator.is_main_process:
                        steps_trained_progress_bar.write("Early stopping triggered. Stopping training.")
                    continue_training = False
                    break
"""
        # Find the end of evaluation block to insert logic
        # We insert it after log_metric(..., prefix="eval")
        insertion_point = 'prefix="eval",\n                )'
        if insertion_point in script_content:
            script_content = script_content.replace(insertion_point, insertion_point + early_stopping_logic)

    training_script_path.write_text(script_content)

    # -------------------------
    # TRAINING COMMAND
    # -------------------------
    model_name = "parler-tts/parler-tts-mini-v1"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Optimized Hyperparameters:
    # - max_steps: 800 (as requested)
    # - learning_rate: 5e-5 (standard for fine-tuning)
    # - lr_scheduler_type: "cosine" (better convergence)
    # - warmup_steps: 50 (smooth start)
    # - evaluation_strategy: "steps" (needed for early stopping)
    # - eval_steps: 100 (evaluate every 100 steps)
    # - save_steps: 100 (save every 100 steps)
    # - weight_decay: 0.01 (regularization)
    
    training_command = f"""
accelerate launch --num_processes={NUM_GPUS} training/run_parler_tts_training.py \\
  --model_name_or_path "{model_name}" \\
  --train_dataset_name "{HF_DATASET_REPO}" \\
  --train_dataset_config_name "default" \\
  --train_split_name "train" \\
  --eval_dataset_name "{HF_DATASET_REPO}" \\
  --eval_dataset_config_name "default" \\
  --eval_split_name "validation" \\
  --max_train_samples 9000 \\
  --max_eval_samples 500 \\
  --seed 42 \\
  --do_train true \\
  --do_eval true \\
  --preprocessing_num_workers 1 \\
  --evaluation_strategy "steps" \\
  --eval_steps 100 \\
  --save_steps 100 \\
  --description_column_name "text_description" \\
  --prompt_column_name "text" \\
  --target_audio_column_name "audio" \\
  --description_tokenizer_name "google/flan-t5-base" \\
  --prompt_tokenizer_name "google/flan-t5-base" \\
  --save_to_disk "/tmp/parler_dataset_processed" \\
  --temporary_save_to_disk "/tmp/parler_dataset_temp" \\
  --output_dir "{OUTPUT_DIR}" \\
  --overwrite_output_dir true \\
  --per_device_train_batch_size 4 \\
  --per_device_eval_batch_size 4 \\
  --gradient_accumulation_steps 4 \\
  --gradient_checkpointing true \\
  --optim "adamw_bnb_8bit" \\
  --learning_rate 5e-5 \\
  --lr_scheduler_type "cosine" \\
  --warmup_steps 50 \\
  --weight_decay 0.01 \\
  --max_steps 800 \\
  --bf16 true \\
  --report_to "none"
"""

    print("\nStarting Parler-TTS fine-tuning on H100…")
    subprocess.run(training_command, shell=True, check=True, cwd=str(repo_path))

    modal.Volume.from_name(VOLUME_NAME).commit()
    print("\nFine-tuning complete!")

# -------------------------
# ENTRYPOINT
# -------------------------
@app.local_entrypoint()
def main():
    finetune_parler_tts.remote()
