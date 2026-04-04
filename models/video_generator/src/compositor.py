from __future__ import annotations
import io
import tempfile
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def remove_background(image_bytes: bytes) -> Image.Image:
    try:
        from rembg import remove as rembg_remove
        result = rembg_remove(image_bytes)
        return Image.open(io.BytesIO(result)).convert("RGBA")
    except Exception as e:
        print(f"  rembg failed ({e}), falling back to threshold removal")
        img  = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        data = np.array(img, dtype=np.uint8)
        r, g, b = data[..., 0], data[..., 1], data[..., 2]
        white_mask = (r >= 240) & (g >= 240) & (b >= 240)
        data[white_mask, 3] = 0
        return Image.fromarray(data, "RGBA")


def composite_frame(
    background_bytes: bytes,
    characters: list[dict],          
    output_size: tuple[int, int],   
) -> bytes:
    """
    Composite background + character PNGs into one frame.

    Each character has:
      - image_bytes: PNG bytes (white background will be removed)
      - position: {x: float, y: float} — 0.0=left/top, 1.0=right/bottom
                  position refers to the CENTER of the character

    Returns PNG bytes of the composited frame.
    """
    W, H = output_size

    bg = Image.open(io.BytesIO(background_bytes)).convert("RGBA")
    bg = bg.resize((W, H), Image.LANCZOS)
    canvas = bg.copy()

    n_chars = len(characters)
    for idx, char in enumerate(characters):
        char_img = remove_background(char["image_bytes"])
        position = char.get("position", {"x": 0.5, "y": 0.5})
        is_speaker = char.get("is_speaker", idx == 0)

        cx = float(position.get("x", 0.5))
        cy = float(position.get("y", 0.5))

        if n_chars == 1:
            char_h = int(H * 0.75)
        elif is_speaker:
            char_h = int(H * 0.70)
        else:
            char_h = int(H * 0.50)

        ratio    = char_h / char_img.height
        char_w   = int(char_img.width * ratio)
        char_img = char_img.resize((char_w, char_h), Image.LANCZOS)

        paste_x = int(cx * W) - char_w // 2
        paste_y = int(cy * H) - char_h // 2

        paste_x = max(0, min(paste_x, W - char_w))
        paste_y = max(0, min(paste_y, H - char_h))

        canvas.paste(char_img, (paste_x, paste_y), mask=char_img.split()[3])
        print(f"  Placed {char.get('name','?')} ({'speaker' if is_speaker else 'listener'}) at ({cx:.2f},{cy:.2f}) → {char_w}x{char_h}px")

    result = canvas.convert("RGB")
    buf    = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def extract_last_frame(video_bytes: bytes) -> bytes:
    """
    Extract the very last frame of a video clip and return as PNG bytes.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        video_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        frame_path = f.name

    subprocess.run([
        "ffmpeg", "-y",
        "-sseof", "-1.5",
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "1",
        frame_path,
    ], check=True, capture_output=True)

    size = Path(frame_path).stat().st_size
    print(f"Last frame extracted ({size/1024:.1f} KB)")
    if size < 1000:
        raise RuntimeError(f"Extracted frame too small ({size} bytes)")

    frame_bytes = Path(frame_path).read_bytes()
    Path(video_path).unlink(missing_ok=True)
    Path(frame_path).unlink(missing_ok=True)
    return frame_bytes