import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from gabor_face import manjunath_bank, magnitude_maps


def _font(size):
    try:
        return ImageFont.load_default(size=size)       # Pillow >= 10.1
    except Exception:
        return ImageFont.load_default()


def draw_detections(pil_img, boxes, raw=None):
    """Return a copy of the image with face boxes (green, with score) and optional raw windows (yellow)."""
    img = pil_img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    lw = max(2, round(max(img.size) / 300))
    font = _font(max(12, round(max(img.size) / 45)))
    for x, y, w, h, _ in (raw or []):
        d.rectangle([x, y, x + w, y + h], outline=(255, 200, 0), width=1)
    for x, y, w, h, p in boxes:
        d.rectangle([x, y, x + w, y + h], outline=(0, 230, 80), width=lw)
        label = f"{p:.2f}"
        tw = d.textlength(label, font=font)
        th = getattr(font, "size", 12) + 4
        d.rectangle([x, max(0, y - th), x + tw + 6, max(0, y - th) + th], fill=(0, 150, 50))
        d.text((x + 3, max(0, y - th) + 1), label, fill=(255, 255, 255), font=font)
    return img


def bank_image(S=4, K=6, tile=110):
    """Real part of the Manjunath & Ma Gabor wavelet family: S scales (rows) x K orientations (columns)."""
    bank = manjunath_bank(S, K)
    size = max(k.shape[0] for row in bank for k in row)
    rows = []
    for row in bank:
        tiles = []
        for k in row:
            canvas = np.zeros((size, size))
            o = (size - k.shape[0]) // 2
            canvas[o:o + k.shape[0], o:o + k.shape[1]] = k.real
            m = np.abs(canvas).max() + 1e-12
            t = ((canvas / m + 1) / 2 * 255).astype(np.uint8)
            t = cv2.resize(t, (tile, tile), interpolation=cv2.INTER_AREA)
            t = cv2.cvtColor(cv2.applyColorMap(t, cv2.COLORMAP_BONE), cv2.COLOR_BGR2RGB)
            tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        rows.append(np.hstack(tiles))
    return Image.fromarray(np.vstack(rows))


def response_image(gray, kernels, width=300):
    """Input image + the four Gabor magnitude maps (0, 45, 90, 135 degrees) side by side."""
    h, w = gray.shape
    s = width / float(max(h, w))
    g = cv2.resize(gray.astype(np.float32), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    tiles = [(np.clip(g, 0, 1) * 255).astype(np.uint8)]
    for m in magnitude_maps(g, kernels):
        tiles.append((np.clip(m / (np.percentile(m, 99.5) + 1e-9), 0, 1) * 255).astype(np.uint8))
    tiles = [cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255) for t in tiles]
    return Image.fromarray(np.hstack(tiles)).convert("RGB")
