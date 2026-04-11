from __future__ import annotations
import io
import os
import subprocess
import tempfile
import json

import numpy as np
from PIL import Image, ImageOps
import imageio

from src.config import Config

DISTILLED_SIGMA_VALUES = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875
]

AUTO_RESOLUTIONS = {
    "16:9":  ((768,  512), (1280,  768), (1920, 1088)),
    "9:16":  ((512,  768), ( 768, 1280), (1088, 1920)),
    "1:1":   ((512,  512), ( 768,  768), (1024, 1024)),
    "4:3":   ((512,  384), ( 768,  576), (1024,  768)),
    "3:4":   ((384,  512), ( 576,  768), ( 768, 1024)),
}

RATIO_VALUES = {
    "16:9": 16/9, "9:16": 9/16, "1:1": 1.0, "4:3": 4/3, "3:4": 3/4,
}

QUALITY_INDEX = {"sd": 0, "hd": 1, "fhd": 2}


def auto_resolution(image: Image.Image, quality: str = "fhd") -> tuple[int, int]:
    w, h    = image.size
    ratio   = w / h
    closest = min(RATIO_VALUES, key=lambda k: abs(RATIO_VALUES[k] - ratio))
    idx     = QUALITY_INDEX.get(quality, 2)
    tw, th  = AUTO_RESOLUTIONS[closest][idx]
    print(f"Image {w}x{h} ({ratio:.3f}) → {closest} → {tw}x{th}")
    return tw, th


def fit_image(image: Image.Image, w: int, h: int) -> Image.Image:
    return ImageOps.fit(image, (w, h), method=Image.LANCZOS, centering=(0.5, 0.5))




def pad_audio(audio_bytes: bytes, pad_secs: float) -> tuple[str, float]:
    suffix = ".mp3" if audio_bytes[:3] == b"ID3" or audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2") else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        raw_path = f.name

    probe = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", raw_path,
    ], capture_output=True, text=True, check=True)
    duration = float(json.loads(probe.stdout)["streams"][0]["duration"])

    padded_path = raw_path.replace(suffix, "_padded.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", raw_path,
        "-af", f"apad=pad_dur={pad_secs}",
        "-c:a", "pcm_s16le",
        padded_path,
    ], check=True, capture_output=True)

    total_duration = duration + pad_secs
    print(f"Audio: {duration:.2f}s + {pad_secs}s pad = {total_duration:.2f}s → {padded_path}")
    os.unlink(raw_path)
    return padded_path, total_duration




def trim_tail(video_bytes: bytes, trim_secs: float) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        inp = f.name
    probe = subprocess.run([
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", inp,
    ], capture_output=True, text=True, check=True)
    duration    = float(json.loads(probe.stdout)["streams"][0]["duration"])
    keep        = max(0.5, duration - trim_secs)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out = f.name
    subprocess.run([
        "ffmpeg", "-y", "-i", inp,
        "-t", str(keep), "-c", "copy", out,
    ], check=True, capture_output=True)
    result = open(out, "rb").read()
    from pathlib import Path as _P
    _P(inp).unlink(missing_ok=True)
    _P(out).unlink(missing_ok=True)
    print(f"Trimmed {trim_secs}s → {keep:.2f}s")
    return result


def calc_num_frames(duration: float, fps: float) -> int:
    total = int(duration * fps)
    return max(round(total / 8) * 8 + 1, 9)


class VideoGenerator:

    def __init__(self, cfg: Config):
        self.cfg  = cfg
        self.pipe = None

    def load_model(self):
        import torch
        from diffusers import DiffusionPipeline

        m        = self.cfg.model
        hf_token = os.environ.get("HF_TOKEN")
        print("Loading pipeline...")
        self.pipe = DiffusionPipeline.from_pretrained(
            m.id,
            custom_pipeline=m.pipeline,
            torch_dtype=torch.bfloat16,
            cache_dir=m.cache_dir,
            token=hf_token,
        )
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        print("Loading Camera Control LoRA...")
        self.pipe.load_lora_weights(
            m.lora_id,
            adapter_name="camera_control",
            cache_dir=m.cache_dir,
            token=hf_token,
        )
        self.pipe.fuse_lora(lora_scale=m.lora_scale)
        self.pipe.unload_lora_weights()
        self.pipe.to("cuda")
        print("Model ready.")

    def _infer(
        self,
        image_bytes:  bytes,
        audio_bytes:  bytes,
        prompt:       str,
        negative_prompt: str,
        seed:         int,
        tail_secs:    float = 1.5,
        quality:      str   = "fhd",
    ) -> bytes:
        import torch

        if seed == -1:
            import random
            seed = random.randint(0, 1_000_000)

        raw    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = auto_resolution(raw, quality)
        image  = fit_image(raw, width, height)

        # The Static Camera LoRA responds best when camera control tokens
        # appear at the START and are the dominant instruction.
        # User prompt is appended after — animation details only, no camera words.
        STATIC_PREFIX = (
            "Static camera. Fixed wide shot. Camera locked. No zoom. No movement. "
            "No dolly. No pan. No tilt. No camera shake. "
        )
        STATIC_SUFFIX = (
            " Camera does not move. Static shot. Wide angle. No zoom in. No close up."
        )
        prompt = STATIC_PREFIX + prompt.strip() + STATIC_SUFFIX

        padded_audio, total_duration = pad_audio(audio_bytes, pad_secs=tail_secs)
        num_frames = calc_num_frames(total_duration, self.cfg.generation.fps)
        print(f"Prompt: {prompt[:120]}...")
        print(f"Res: {width}x{height} | Frames: {num_frames} | Seed: {seed}")

        torch.cuda.empty_cache()
        generator = torch.Generator("cuda").manual_seed(seed)

        video_output, _ = self.pipe(
            image=image,
            audio=padded_audio,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=self.cfg.generation.fps,
            num_inference_steps=8,
            sigmas=DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,
            generator=generator,
            return_dict=False,
        )

        frames = video_output[0] if isinstance(video_output[0], list) else video_output
        np_frames = [np.array(img) for img in frames]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            raw_path = f.name
        imageio.mimsave(raw_path, np_frames, fps=self.cfg.generation.fps, format="mp4", quality=9)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            muxed_path = f.name
        subprocess.run([
            "ffmpeg", "-y",
            "-i", raw_path, "-i", padded_audio,
            "-c:v", "copy", "-c:a", "aac",
            muxed_path,
        ], check=True, capture_output=True)

        result = open(muxed_path, "rb").read()
        from pathlib import Path as _P
        for p in [padded_audio, raw_path, muxed_path]:
            _P(p).unlink(missing_ok=True)
        return result

    def generate(
        self,
        image_bytes: bytes,
        audio_bytes: bytes,
        prompt:      str | None = None,
        seed:        int = -1,
    ) -> bytes:
        if prompt is None:
            prompt = self.cfg.generation.default_prompt
        clip = self._infer(
            image_bytes=image_bytes,
            audio_bytes=audio_bytes,
            prompt=prompt,
            negative_prompt=self.cfg.negative_prompt,
            seed=seed,
            tail_secs=1.5,
        )
        return trim_tail(clip, trim_secs=1.5)

    def generate_pipeline(
        self,
        shots:             list[dict],
        background_images: dict,
        character_images:  dict,
        audio_files:       dict,
        seed:              int = -1,
    ) -> dict[str, bytes]:
        from src.compositor import composite_frame

        results = {}

        first_bg_name  = shots[0]["background_name"]
        first_bg_bytes = background_images[first_bg_name]
        raw_ref        = Image.open(io.BytesIO(first_bg_bytes)).convert("RGB")
        width, height  = auto_resolution(raw_ref, quality=self.cfg.generation.quality)
        print(f"Output resolution: {width}x{height}")

        RELAX_SECS       = 2.5   # silence after voice ends — character settles
        DARK_BUFFER_SECS = 2.0   # extra dark seconds trimmed before delivery
        TAIL_SECS = DARK_BUFFER_SECS + RELAX_SECS  # 4.5s

        for i, shot in enumerate(shots):
            shot_id   = shot["shot_id"]
            bg_name   = shot["background_name"]
            prompt    = shot.get("video_prompt") or self.cfg.generation.default_prompt
            neg       = shot.get("negative_prompt") or self.cfg.negative_prompt
            scene_num = shot["scene_number"]
            shot_num  = shot["shot_number"]
            speaker   = shot.get("speaker", "")

            print(f"\n{'='*60}")
            print(f"Scene {scene_num} | Shot {shot_num} | {shot_id} | speaker={speaker}")

            bg_bytes    = background_images.get(bg_name)
            scene_chars = shot.get("characters_present", [])

            # ── ALWAYS composite a fresh frame for every shot ──
            if scene_chars:
                print(f"  Compositing fresh frame for shot {shot_id}...")
                chars_in_shot = []
                for cp in scene_chars:
                    name     = cp["name"] if isinstance(cp, dict) else cp
                    position = cp.get("position", {"x": 0.5, "y": 0.5}) if isinstance(cp, dict) else {"x": 0.5, "y": 0.5}
                    c_bytes  = character_images.get(name)
                    if c_bytes is None:
                        print(f"  WARNING: Character not found: {name}, skipping")
                        continue
                    chars_in_shot.append({
                        "name":        name,
                        "image_bytes": c_bytes,
                        "position":    position,
                        "is_speaker":  False,
                    })
                frame_bytes = composite_frame(
                    background_bytes = bg_bytes,
                    characters       = chars_in_shot,
                    output_size      = (width, height),
                )
                print(f"  Frame composited: {len(frame_bytes)/1024:.1f} KB")

                # DEBUG: save composited frame to verify characters look correct before video model
                debug_path = f"debug_composite_scene{scene_num}_shot{shot_num}.png"
                with open(debug_path, "wb") as _df:
                    _df.write(frame_bytes)
                print(f"  DEBUG frame saved → {debug_path}")
            elif bg_bytes is not None:
                print("  WARNING: No characters — using bare background.")
                frame_bytes = bg_bytes
            else:
                raise RuntimeError(f"No starting frame available for shot {shot_id}")

            audio_bytes_shot = audio_files.get(shot_id)
            if audio_bytes_shot is None:
                raise ValueError(f"Audio not found for shot: {shot_id}")

            print(f"  Generating clip for {shot_id}...")
            raw_clip = self._infer(
                image_bytes=frame_bytes,
                audio_bytes=audio_bytes_shot,
                prompt=prompt,
                negative_prompt=neg,
                seed=seed,
                tail_secs=TAIL_SECS,
            )

            results[shot_id] = trim_tail(raw_clip, trim_secs=DARK_BUFFER_SECS + RELAX_SECS)
            print(f"  Clip done: {len(results[shot_id])/1024:.1f} KB")

        print(f"\nPipeline complete. {len(results)} clips generated.")
        return results