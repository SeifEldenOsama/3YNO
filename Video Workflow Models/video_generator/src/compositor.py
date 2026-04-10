from __future__ import annotations
import io
from pathlib import Path

import numpy as np
from PIL import Image


def remove_background(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # If PNG already has transparency, trust it — skip rembg entirely
    alpha = np.array(img.split()[3])
    if alpha.min() < 255:
        print("  Alpha channel detected — skipping background removal")
        return img

    try:
        from rembg import remove as rembg_remove
        result = rembg_remove(image_bytes)
        return Image.open(io.BytesIO(result)).convert("RGBA")
    except Exception as e:
        print(f"  rembg failed ({e}), falling back to edge-flood removal")
        data = np.array(img, dtype=np.uint8)
        r, g, b = data[..., 0], data[..., 1], data[..., 2]
        H, W    = data.shape[:2]
        near_white = (r >= 235) & (g >= 235) & (b >= 235)
        from PIL import ImageFilter
        from collections import deque
        mask_np = np.array(
            Image.fromarray((near_white * 255).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(3))
        ) > 0
        visited   = np.zeros((H, W), dtype=bool)
        border_bg = np.zeros((H, W), dtype=bool)
        queue     = deque()
        for x in range(W):
            if mask_np[0, x]:   queue.append((0, x))
            if mask_np[H-1, x]: queue.append((H-1, x))
        for y in range(H):
            if mask_np[y, 0]:   queue.append((y, 0))
            if mask_np[y, W-1]: queue.append((y, W-1))
        while queue:
            y, x = queue.popleft()
            if visited[y, x]: continue
            visited[y, x]   = True
            border_bg[y, x] = True
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and mask_np[ny, nx]:
                    queue.append((ny, nx))
        data[border_bg, 3] = 0
        return Image.fromarray(data, "RGBA")


def add_character_border(char_img: Image.Image, border_px: int = 4) -> Image.Image:
    """
    Add a thin white outline around the character so the AI model
    perceives it as a distinct foreground object, not background art.
    Without this, composited characters get 'baked in' as painted shadows.
    """
    from PIL import ImageFilter
    r, g, b, a = char_img.split()
    # Expand the alpha mask slightly to create the border shape
    border_mask = a.filter(ImageFilter.MaxFilter(border_px * 2 + 1))
    # Create white border layer
    border_layer = Image.new("RGBA", char_img.size, (255, 255, 255, 0))
    border_layer.putalpha(border_mask)
    # Composite: border behind character
    result = Image.new("RGBA", char_img.size, (0, 0, 0, 0))
    result.paste(border_layer, (0, 0))
    result.paste(char_img, (0, 0), mask=a)
    return result


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

        char_h = int(H * 0.55) 

        ratio    = char_h / char_img.height
        char_w   = int(char_img.width * ratio)
        char_img = char_img.resize((char_w, char_h), Image.LANCZOS)


        paste_x = int(cx * W) - char_w // 2
        paste_y = int(cy * H) - char_h // 2


        paste_x = max(-char_w // 2, min(paste_x, W - char_w // 2))
        paste_y = max(-char_h // 2, min(paste_y, H - char_h // 2))

        char_img = add_character_border(char_img, border_px=4)
        canvas.paste(char_img, (paste_x, paste_y), mask=char_img.split()[3])
        print(f"  Placed {char.get('name','?')} ({'speaker' if is_speaker else 'listener'}) at ({cx:.2f},{cy:.2f}) → {char_w}x{char_h}px")

    result = canvas.convert("RGB")
    buf    = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()