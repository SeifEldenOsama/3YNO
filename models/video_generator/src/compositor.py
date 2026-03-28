from __future__ import annotations
import io
import tempfile
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def remove_white_background(image_bytes: bytes, threshold: int = 240) -> Image.Image:
    """
    Remove white (or near-white) background from a character PNG.
    Returns RGBA image where white pixels become transparent.

    threshold: pixels where R,G,B are all >= threshold are treated as background.
    """
    img  = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    data = np.array(img, dtype=np.uint8)

    r, g, b, a = data[..., 0], data[..., 1], data[..., 2], data[..., 3]
    white_mask = (r >= threshold) & (g >= threshold) & (b >= threshold)

    data[white_mask, 3] = 0          # make white pixels fully transparent
    return Image.fromarray(data, "RGBA")


def composite_frame(
    background_bytes: bytes,
    characters: list[dict],          # [{name, image_bytes, position: {x, y}}]
    output_size: tuple[int, int],    # (width, height) of final frame
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

    # ── Load and resize background ────────────────────────────────────────────
    bg = Image.open(io.BytesIO(background_bytes)).convert("RGBA")
    bg = bg.resize((W, H), Image.LANCZOS)
    canvas = bg.copy()

    # ── Place each character ──────────────────────────────────────────────────
    for char in characters:
        char_img   = remove_white_background(char["image_bytes"])
        position   = char.get("position", {"x": 0.5, "y": 0.5})

        cx = float(position.get("x", 0.5))
        cy = float(position.get("y", 0.5))

        # Scale character to ~30% of canvas height (adjust as needed)
        char_h = int(H * 0.35)
        ratio  = char_h / char_img.height
        char_w = int(char_img.width * ratio)
        char_img = char_img.resize((char_w, char_h), Image.LANCZOS)

        # Position: cx,cy = center of character on canvas
        paste_x = int(cx * W) - char_w // 2
        paste_y = int(cy * H) - char_h // 2

        # Clamp to canvas bounds
        paste_x = max(0, min(paste_x, W - char_w))
        paste_y = max(0, min(paste_y, H - char_h))

        canvas.paste(char_img, (paste_x, paste_y), mask=char_img.split()[3])
        print(f"  Placed {char.get('name','?')} at ({cx:.2f},{cy:.2f}) → pixel ({paste_x},{paste_y})")

    # ── Convert back to RGB and return PNG bytes ──────────────────────────────
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

    # sseof=-0.1 seeks to 0.1s before end and grabs 1 frame
    subprocess.run([
        "ffmpeg", "-y",
        "-sseof", "-0.1",
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
