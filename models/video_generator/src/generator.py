from __future__ import annotations
import glob
import io
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.config import Config


DISTILLED_SIGMA_VALUES = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875
]


def patch_custom_pipeline():
    """Remove `additive_mask=True` kwarg injected by newer diffusers into the
    cached HuggingFace custom pipeline, which older pipeline.py doesn't accept."""
    cache_dir = "/root/.cache/huggingface/modules/diffusers_modules/local/multimodalart--ltx2-audio-to-video"
    pipeline_files = glob.glob(f"{cache_dir}/**/pipeline.py", recursive=True)
    for file_path in pipeline_files:
        print(f"Patching: {file_path}")
        with open(file_path, "r") as f:
            content = f.read()
        new_content = content.replace(", additive_mask=True", "")
        with open(file_path, "w") as f:
            f.write(new_content)
    print("Patching complete.")


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


def prepare_image(image_bytes: bytes, width: int, height: int) -> Image.Image:
    """Load and pre-process the input image."""
    raw = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    print(f"Original image: {raw.size}")
    img = raw.filter(ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=3))
    img = ImageEnhance.Contrast(img).enhance(1.05)
    img = img.resize((width, height), Image.LANCZOS)
    print(f"Resized to: {img.size}")
    return img


def preprocess_audio(audio_bytes: bytes, target_sr: int = 16000) -> tuple[str, float]:
    """Normalise audio to 16 kHz mono PCM-16 WAV. Returns (path, duration_s)."""
    import librosa
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        input_path = f.name

    audio, sr = librosa.load(input_path, sr=target_sr, mono=True)
    os.unlink(input_path)
    duration = len(audio) / sr
    print(f"Audio: {sr}Hz, {duration:.2f}s, shape={audio.shape}")

    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name
    sf.write(out_path, audio, target_sr, subtype="PCM_16")
    print(f"Audio preprocessed → {out_path}")
    return out_path, duration


def calc_num_frames(duration: float, fps: float) -> int:
    """(num_frames - 1) must be divisible by 8."""
    total = int(duration * fps)
    base_block = round(total / 8) * 8
    num_frames = base_block + 1
    return max(num_frames, 9)


def frames_to_video(flat_frames, height: int, width: int, fps: float, temp_vid: str):
    """Write a list of PIL images / tensors to an intermediate MP4."""
    import imageio

    processed = []
    for i, frame in enumerate(flat_frames):
        if isinstance(frame, Image.Image):
            arr = np.array(frame.convert("RGB"))
        else:
            import torch
            if torch.is_tensor(frame):
                arr = frame.cpu().float()
                if arr.ndim == 4:
                    arr = arr.squeeze(0)
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                    arr = arr.permute(1, 2, 0)
                arr = arr.numpy()
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
    print(f"Intermediate video written: {temp_vid}")


def merge_and_encode(temp_vid: str, audio_path: str, final_vid: str):
    """High-quality mux + re-encode with audio."""
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


class VideoGenerator:
    """Generates a video clip from one frame image + one audio file."""

    def __init__(self, cfg: Config):
        self.cfg  = cfg
        self.pipe = None

    def load_model(self):
        import torch
        from diffusers import DiffusionPipeline

        m = self.cfg.model
        hf_token = os.environ.get("HF_TOKEN")
        print("Loading pipeline...")
        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                m.id,
                custom_pipeline=m.pipeline,
                torch_dtype=torch.bfloat16,
                cache_dir=m.cache_dir,
                token=hf_token,
            )
        except TypeError:
            patch_custom_pipeline()
            self.pipe = DiffusionPipeline.from_pretrained(
                m.id,
                custom_pipeline=m.pipeline,
                torch_dtype=torch.bfloat16,
                cache_dir=m.cache_dir,
                token=hf_token,
            )

        print(f"Loading Camera Control LoRA: {m.lora_id}")
        self.pipe.load_lora_weights(
            m.lora_id,
            adapter_name="camera_control",
            cache_dir=m.cache_dir,
            token=hf_token,
        )
        self.pipe.fuse_lora(lora_scale=m.lora_scale)
        self.pipe.unload_lora_weights()
        print("LoRA fused.")

        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        self.pipe.to("cuda")
        print("Model ready.")

    def generate(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        prompt:      str | None = None,
        seed:        int = 42,
    ) -> bytes:
        import torch

        g = self.cfg.generation

        width, height  = get_resolution(image_bytes)
        image          = prepare_image(image_bytes, width, height)
        audio_path, audio_duration = preprocess_audio(audio_bytes, target_sr=16000)

        fps            = g.fps
        video_duration = min(audio_duration, g.max_duration)
        num_frames     = calc_num_frames(video_duration, fps)
        print(f"Duration: {video_duration:.2f}s → num_frames={num_frames}")

        if prompt is None:
            prompt = g.default_prompt

        negative_prompt = self.cfg.negative_prompt
        print(f"Prompt: {prompt}")
        print(f"Res: {width}x{height} | Frames: {num_frames} | FPS: {fps} | Seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        with torch.no_grad():
            video_output = self.pipe(
                image=image,
                audio=audio_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=fps,
                num_inference_steps=g.num_steps,
                sigmas=DISTILLED_SIGMA_VALUES,
                guidance_scale=g.guidance_scale,
                generator=generator,
                return_dict=False,
            )

        raw_frames_out = video_output[0] if isinstance(video_output, (list, tuple)) else video_output
        print(f"Output type: {type(raw_frames_out)}, len: {len(raw_frames_out)}")

        flat_frames = []
        for item in raw_frames_out:
            if isinstance(item, list):
                flat_frames.extend(item)
            else:
                flat_frames.append(item)
        print(f"Total flat frames: {len(flat_frames)}")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_vid = f.name
        final_vid = temp_vid.replace(".mp4", "_final.mp4")

        frames_to_video(flat_frames, height, width, fps, temp_vid)

        try:
            merge_and_encode(temp_vid, audio_path, final_vid)
            output_path = final_vid
        except Exception as e:
            print(f"HQ encode failed: {e}, trying fallback...")
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", temp_vid, "-i", audio_path,
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", final_vid,
                ], check=True, capture_output=True)
                output_path = final_vid
            except Exception as e2:
                print(f"All merges failed: {e2}. Silent video.")
                output_path = temp_vid

        result = open(output_path, "rb").read()
        for p in [audio_path, temp_vid, final_vid]:
            if os.path.exists(p):
                os.unlink(p)
        return result
