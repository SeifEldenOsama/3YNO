import modal
import subprocess
import shutil
from pathlib import Path

app = modal.App("download-tts-fixed")

volume = modal.Volume.from_name("tts-dataset-storage", create_if_missing=True)

@app.function(
    image=modal.Image.debian_slim().pip_install("kaggle"),
    secrets=[modal.Secret.from_name("kaggle-secret")],
    volumes={"/data": volume},
    timeout=3600,
)
def download_to_volume():
    """Download Kaggle TTS dataset and structure it correctly"""

    target_dir = Path("/data/tts_dataset/teacher_dataset_large_updated")
    voices_dir = target_dir / "voices"

    target_dir.mkdir(parents=True, exist_ok=True)
    voices_dir.mkdir(exist_ok=True)

    temp_download_path = Path("/tmp/kaggle_data")
    temp_download_path.mkdir(exist_ok=True)

    print(f"Downloading Kaggle dataset to {temp_download_path}...")

    cmd = [
        "kaggle", "datasets", "download",
        "-d", "seifosamahosney/tts-dataset",
        "-p", str(temp_download_path),
        "--unzip",
        "--force",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Kaggle Error:\n{result.stderr}")
        return "Download failed"

    print("Download successful! Locating dataset files...")

    voices_dirs = list(temp_download_path.rglob("voices"))

    if not voices_dirs:
        print("voices folder not found anywhere!")
        print("Extracted contents:")
        for p in temp_download_path.rglob("*"):
            print(" -", p)
        return "voices folder missing"

    source_voices_dir = voices_dirs[0]
    print(f"Found voices folder at: {source_voices_dir}")

    count = 0
    for wav_file in source_voices_dir.rglob("*.wav"):
        target_file = voices_dir / wav_file.name
        if not target_file.exists():
            shutil.move(str(wav_file), str(target_file))
            count += 1

    print(f"Moved {count} .wav files to {voices_dir}")

    metadata_files = list(temp_download_path.rglob("metadata.jsonl"))
    metadata_dst = target_dir / "metadata.jsonl"

    if metadata_files:
        shutil.move(str(metadata_files[0]), str(metadata_dst))
        print(f"Moved metadata.jsonl to {metadata_dst}")
    else:
        print("metadata.jsonl not found")

    has_wavs = any(voices_dir.glob("*.wav"))
    has_metadata = metadata_dst.exists()

    print(f"WAV files present: {has_wavs}")
    print(f"Metadata present: {has_metadata}")

    volume.commit()
    print("Saved to permanent volume 'tts-dataset-storage'")

    return f"Dataset ready at {target_dir}"

@app.local_entrypoint()
def main():
    download_to_volume.remote()
