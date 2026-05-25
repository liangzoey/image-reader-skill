---
name: image-reader
description: >
  Automatically read and analyze images using local OCR (EasyOCR + OpenCV) and/or
  Janus-Pro (local multimodal VLM) when the current model cannot process images
  directly. Auto-detects GPU/VRAM/RAM and picks the best analysis mode. Use this
  skill whenever the user provides a path to an image file (.png, .jpg, .jpeg, .bmp,
  .webp) and asks you to "read", "look at", "analyze", "describe", or extract
  information from the image.
---

# Image Reader

## Main Entry Point

```
python scripts/analyze.py <image_path> [prompt] 2>$null
```

Auto-detects hardware and runs the best available mode.

## Auto-Detect Logic

| VRAM | GPU | Mode |
|------|-----|------|
| >=14GB | Yes | Janus-Pro-7B (FP16) |
| >=8GB | Yes | Janus-Pro-7B (4-bit) or 1B fallback |
| >=4GB | Yes | Janus-Pro-1B |
| <4GB or no GPU | Yes/No | EasyOCR only |
| No GPU, >=16GB RAM | No | Janus-Pro-1B on CPU |

Model cache is checked before loading. If not cached, returns a download command instead of hanging.

## Scripts

### `scripts/analyze.py` (recommended)
- Auto-detects GPU/VRAM/RAM
- Runs OCR for structural data + Janus-Pro for deep understanding
- Checks model cache before attempting download
- `--use-small` : force 1B model
- `--mode ocr` : OCR only
- `--model-path <path>` : local model dir

### `scripts/ocr_image.py`
- EasyOCR text extraction (Chinese + English)
- OpenCV structural analysis: 4x4 grid, edge density, type classification
- ~5s on GPU

### `scripts/janus_analyze.py`
- Janus-Pro-7B or 1B for deep image understanding
- Auto-downloads from HuggingFace on first use
- `--use-small` : 1B model (~2GB)
- `--model-path <path>` : local model override
- Requires ~14GB VRAM (7B) or ~3GB (1B)

### `scripts/auto_detect.py`
- Standalone hardware info tool
- Outputs GPU name, VRAM, RAM, and recommended mode

### `janus/` (cloned Janus-Pro source)
Used by janus_analyze.py for model inference.

## Examples

User provides image path:
1. Run `python scripts/analyze.py "C:\path\to\image.png" 2>$null`
2. Parse JSON: hardware info, OCR text, structure grid, Janus description
3. Present merged summary to the user
