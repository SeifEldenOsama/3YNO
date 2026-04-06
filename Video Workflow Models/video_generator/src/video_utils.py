from __future__ import annotations
import io
import os
import json
import glob
import tempfile
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter



DISTILLED_SIGMA_VALUES = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875
]



def patch_custom_pipeline():
    """
    Remove ', additive_mask=True' from the cached custom pipeline file.
    This fixes the TypeError: LTX2TextConnectors.forward() got an unexpected
    keyword argument 'additive_mask' when using transformers==4.52.4.
    """
    cache_dir = "/root/.cache/huggingface/modules/diffusers_modules/local/multimodalart--ltx2-audio-to-video"
    pipeline_files = glob.glob(f"{cache_dir}/**/pipeline.py", recursive=True)
    for file_path in pipeline_files:
        print(f"Patching: {file_path}")
        with open(file_path, "r") as f:
            content = f.read()
        patched = content.replace(", additive_mask=True", "")
        with open(file_path, "w") as f:
            f.write(patched)
    print("Pipeline patched.")


def get_resolution(image_bytes: bytes) -> tuple[int, int]:
    """Pick resolution matching image aspect ratio."""
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    ratio = w / h
    resolutions = {
        "1:1":  (512, 512,  1.0),
        "16:9": (768, 512,  16/9),
        "9:16": (512, 768,  9/16),
    }
    best = min(resolutions.values(), key=lambda r: abs(r[2] - ratio))
    print(f"Image ratio {ratio:.3f} → resolution {best[0]}x{best[1]}")
    return best[0], best[1]


def calc_num_frames(duration: float, fps: float) -> int:
    """(num_frames - 1) must be divisible by 8."""
    total      = int(duration * fps)
    base_block = round(total / 8) * 8
    num_frames = base_block + 1
    return max(num_frames, 9)



def prepare_image(image_bytes: bytes, width: int, height: int) -> Image.Image:
    raw = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    print(f"Original image: {raw.size}")
    img = raw.filter(ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=3))
    img = ImageEnhance.Contrast(img).enhance(1.05)
    img = img.resize((width, height), Image.LANCZOS)
    print(f"Prepared image: {img.size}")
    return img


def preprocess_audio(audio_bytes: bytes, target_sr: int = 16000) -> tuple[str, float]:
    """
    Resample audio to target_sr, normalize, and save to temp WAV.
    Returns (temp_path, duration_seconds).
    """
    import librosa
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        input_path = f.name

    audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
    os.unlink(input_path)

    duration = len(audio) / sr
    print(f"Audio: {sr}Hz, {duration:.2f}s")

    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name
    sf.write(out_path, audio, target_sr, subtype="PCM_16")
    print(f"Audio preprocessed → {out_path}")
    return out_path, duration


def frames_to_video(flat_frames: list, height: int, width: int,
                    fps: float, temp_vid: str):
    """Convert list of frames (tensor or PIL) to an mp4 file."""
    import torch
    import imageio

    processed = []
    for i, frame in enumerate(flat_frames):
        if torch.is_tensor(frame):
            arr = frame.cpu().float()
            if arr.ndim == 4:
                arr = arr.squeeze(0)
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = arr.permute(1, 2, 0)
            arr = arr.numpy()
        elif isinstance(frame, Image.Image):
            arr = np.array(frame.convert("RGB"))
        else:
            arr = np.array(frame)

        if arr.dtype != np.uint8:
            if arr.max() <= 1.1:
                arr = (arr * 255).clip(0, 255)
            arr = arr.astype(np.uint8)

        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        elif arr.shape[-1] == 1:
            arr = np.concatenate([arr, arr, arr], axis=-1)

        assert arr.shape == (height, width, 3), f"Frame {i} bad shape: {arr.shape}"
        processed.append(arr)

    np_frames = np.stack(processed)
    print(f"Frames shape: {np_frames.shape}")

    writer = imageio.get_writer(
        temp_vid, fps=fps, format="FFMPEG",
        codec="libx264", quality=10, pixelformat="yuv420p",
    )
    for frame in np_frames:
        writer.append_data(frame)
    writer.close()
    print(f"Intermediate video: {temp_vid}")


def merge_and_encode(temp_vid: str, audio_path: str, final_vid: str):
    """High quality ffmpeg mux of video + audio."""
    subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_vid, "-i", audio_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-crf", "16", "-preset", "slow",
        "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", "-shortest",
        final_vid,
    ], check=True, capture_output=True)
    print(f"Final video: {final_vid}")
