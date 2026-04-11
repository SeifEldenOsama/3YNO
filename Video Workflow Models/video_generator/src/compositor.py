from __future__ import annotations
import io
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

        # Keep characters large — clamping below will shift position inward if needed
        if n_chars == 1:
            scale = 0.55
        elif n_chars == 2:
            scale = 0.55
        elif n_chars == 3:
            scale = 0.45
        else:
            scale = 0.38
        char_h = int(H * scale)

        ratio    = char_h / char_img.height
        char_w   = int(char_img.width * ratio)
        char_img = char_img.resize((char_w, char_h), Image.LANCZOS)

        paste_x = int(cx * W) - char_w // 2
        paste_y = int(cy * H) - char_h // 2

        # Clamp so the full character stays inside the frame
        paste_x = max(0, min(paste_x, W - char_w))
        paste_y = max(0, min(paste_y, H - char_h))

        canvas.paste(char_img, (paste_x, paste_y), mask=char_img.split()[3])
        print(f"  Placed {char.get('name','?')} ({'speaker' if is_speaker else 'listener'}) at ({cx:.2f},{cy:.2f}) → {char_w}x{char_h}px")

    result = canvas.convert("RGB")
    buf    = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()