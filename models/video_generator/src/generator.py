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
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    ratio = w / h
    resolutions = {
        "1:1":   (768,  768,  1.0),
        "16:9":  (1280, 720,  16/9),
        "9:16":  (720,  1280, 9/16),
        "4:3":   (768,  576,  4/3),
        "3:4":   (576,  768,  3/4),
    }
    best = min(resolutions.values(), key=lambda r: abs(r[2] - ratio))
    print(f"Image ratio {ratio:.3f} → resolution {best[0]}x{best[1]}")
    return best[0], best[1]

def prepare_image(image_bytes: bytes, width: int, height: int) -> Image.Image:
    raw = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    print(f"Original image: {raw.size}")
    if raw.width < width or raw.height < height:
        raw = raw.resize(
            (max(raw.width, width * 2), max(raw.height, height * 2)),
            Image.LANCZOS,
        )
    img = raw.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=2))
    img = ImageEnhance.Contrast(img).enhance(1.08)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    img = img.resize((width, height), Image.LANCZOS)
    print(f"Prepared image: {img.size}")
    return img

def preprocess_audio(audio_bytes: bytes, target_sr: int = 16000) -> tuple[str, float]:
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
    total = int(duration * fps)
    base_block = round(total / 8) * 8
    num_frames = base_block + 1
    return max(num_frames, 9)

def frames_to_video(flat_frames, height: int, width: int, fps: float, temp_vid: str):
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
    subprocess.run([
        "ffmpeg", "-y",
        "-i", temp_vid, "-i", audio_path,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-crf", "10", "-preset", "veryslow",
        "-profile:v", "high", "-level", "4.2",
        "-pix_fmt", "yuv420p",
        "-tune", "animation",
        "-x264-params", "ref=6:bframes=8:me=umh:subme=10:trellis=2",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
        "-movflags", "+faststart", "-shortest",
        final_vid,
    ], check=True, capture_output=True)
    print(f"Final video: {final_vid}")


def extend_clip(video_path: str, relax_secs: float, dump_secs: float, fps: float) -> str:
    total_extra = relax_secs + dump_secs

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        frame_path = f.name
    subprocess.run([
        "ffmpeg", "-y",
        "-sseof", "-0.1", "-i", video_path,
        "-vframes", "1", "-q:v", "1",
        frame_path,
    ], check=True, capture_output=True)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        frozen_path = f.name
    subprocess.run([
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", str(fps), "-i", frame_path,
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-t", str(total_extra),
        "-c:v", "libx264", "-crf", "10", "-preset", "veryslow",
        "-pix_fmt", "yuv420p", "-r", str(fps),
        "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "320k", "-ar", "48000",
        frozen_path,
    ], check=True, capture_output=True)

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write(f"file '{video_path}'\nfile '{frozen_path}'\n")
        concat_list = f.name

    extended_path = video_path.replace(".mp4", "_ext.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        extended_path,
    ], check=True, capture_output=True)

    for p in [frame_path, frozen_path, concat_list]:
        from pathlib import Path as _P
        _P(p).unlink(missing_ok=True)

    print(f"Clip extended: +{relax_secs}s relax +{dump_secs}s dump → {extended_path}")
    return extended_path


def trim_clip(video_bytes: bytes, trim_secs: float) -> bytes:
    from pathlib import Path as _P
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        input_path = f.name

    probe = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", input_path,
    ], capture_output=True, text=True)
    duration    = float(probe.stdout.strip())
    new_duration = max(0.0, duration - trim_secs)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        output_path = f.name

    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-t", str(new_duration),
        "-c", "copy",
        output_path,
    ], check=True, capture_output=True)

    trimmed = open(output_path, "rb").read()
    _P(input_path).unlink(missing_ok=True)
    _P(output_path).unlink(missing_ok=True)
    print(f"Clip trimmed: -{trim_secs}s dump removed ({duration:.2f}s → {new_duration:.2f}s)")
    return trimmed

class VideoGenerator:

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

        extended_path = extend_clip(output_path, relax_secs=1.0, dump_secs=3.0, fps=fps)

        result = open(extended_path, "rb").read()
        for p in [audio_path, temp_vid, final_vid, extended_path]:
            if os.path.exists(p):
                os.unlink(p)
        return result

    def generate_pipeline(
        self,
        shots:             list[dict],   # from video_timeline.json["shots"]
        background_images: dict,
        character_images:  dict,
        audio_files:       dict,
        seed:              int = 42,
    ) -> dict[str, bytes]:
        from src.compositor import composite_frame, extract_last_frame

        results      = {}
        prev_clip    = None

        first_bg_name  = shots[0]["background_name"]
        first_bg_bytes = background_images[first_bg_name]
        width, height  = get_resolution(first_bg_bytes)
        print(f"Output resolution: {width}x{height}")

        for i, shot in enumerate(shots):
            shot_id      = shot["shot_id"]
            bg_name      = shot["background_name"]
            frame_source = shot.get("frame_source", "composite")
            prompt       = shot.get("video_prompt", None)
            scene_num    = shot["scene_number"]
            shot_num     = shot["shot_number"]

            print(f"\n{'='*60}")
            print(f"Scene {scene_num} | Shot {shot_num} | {shot_id} | source={frame_source}")

            if frame_source == "composite" or prev_clip is None:
                print("  Compositing frame from background + characters...")

                bg_bytes = background_images.get(bg_name)
                if bg_bytes is None:
                    raise ValueError(f"Background not found: {bg_name}")

                chars_in_shot = []

                if shot.get("speaker"):
                    name     = shot["speaker"]
                    position = shot.get("speaker_position", {"x": 0.5, "y": 0.5})
                    c_bytes  = character_images.get(name)
                    if c_bytes is None:
                        print(f"  WARNING: Character image not found: {name}, skipping")
                    else:
                        chars_in_shot.append({
                            "name":        name,
                            "image_bytes": c_bytes,
                            "position":    position,
                        })
                else:
                    for cp in shot.get("characters_present", []):
                        name     = cp["name"] if isinstance(cp, dict) else cp
                        position = cp.get("position", {"x": 0.5, "y": 0.5}) if isinstance(cp, dict) else {"x": 0.5, "y": 0.5}
                        c_bytes  = character_images.get(name)
                        if c_bytes is None:
                            print(f"  WARNING: Character image not found: {name}, skipping")
                            continue
                        chars_in_shot.append({
                            "name":        name,
                            "image_bytes": c_bytes,
                            "position":    position,
                        })

                frame_bytes = composite_frame(
                    background_bytes = bg_bytes,
                    characters       = chars_in_shot,
                    output_size      = (width, height),
                )
                print(f"  Frame composited: {len(frame_bytes)/1024:.1f} KB")

            else:
                print("  Extracting last frame from previous clip...")
                frame_bytes = extract_last_frame(prev_clip)
                print(f"  Last frame: {len(frame_bytes)/1024:.1f} KB")

            audio_bytes = audio_files.get(shot_id)
            if audio_bytes is None:
                raise ValueError(f"Audio not found for shot: {shot_id}")

            print(f"  Generating clip for {shot_id}...")
            clip_bytes = self.generate(
                image_bytes = frame_bytes,
                audio_bytes = audio_bytes,
                prompt      = prompt,
                seed        = seed,
            )

            prev_clip           = clip_bytes
            results[shot_id]    = trim_clip(clip_bytes, trim_secs=3.0)
            print(f"  Clip done: {len(results[shot_id])/1024:.1f} KB (trimmed)")

        print(f"\nPipeline complete. {len(results)} clips generated.")
        return results