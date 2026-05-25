---
name: image-reader
description: >
  Automatically read and analyze images (or video) using local OCR (EasyOCR + OpenCV),
  Janus-Pro (local multimodal VLM), or Qwen2.5-VL GGUF models when the current model
  cannot process images or video directly. Auto-detects GPU/VRAM/RAM and picks the best
  analysis mode. Qwen GGUF mode also supports video analysis (frame extraction + VLM).
  Use this skill whenever the user provides a path to an image file (.png, .jpg, .jpeg,
  .bmp, .webp), a video file (.mp4, .avi, .mov, .mkv, .webm), and asks you to "read",
  "look at", "analyze", "describe", or extract information from the media.
---

# Image Reader

## Main Entry Point

For images:
```
python scripts/analyze.py <image_path> [prompt] 2>$null
```

For video (requires Qwen GGUF model):
```
python scripts/analyze.py <video_path> [prompt] --mode qwen --qwen-model-path <gguf_dir> 2>$null
```

Auto-detects hardware and runs the best available mode.

## Auto-Detect Logic

| Condition | Mode |
|-----------|------|
| Qwen GGUF model found + >=20GB VRAM | Qwen2.5-VL-72B (GGUF) — images + video |
| Qwen GGUF model found + >=6GB VRAM | Qwen2.5-VL-7B (GGUF) — images + video |
| Qwen GGUF model found + >=16GB RAM (no GPU) | Qwen2.5-VL-7B on CPU (GGUF) |
| >=14GB VRAM (no Qwen GGUF) | Janus-Pro-7B (FP16) |
| >=8GB VRAM (no Qwen GGUF) | Janus-Pro-7B (4-bit) or 1B fallback |
| >=4GB VRAM (no Qwen GGUF) | Janus-Pro-1B |
| <4GB or no GPU | EasyOCR only |
| No GPU, >=16GB RAM (no Qwen GGUF) | Janus-Pro-1B on CPU |

Model cache is checked before loading. If not cached, returns a download command instead of hanging.

## Scripts

### `scripts/analyze.py` (recommended)
- Auto-detects GPU/VRAM/RAM and Qwen GGUF model availability
- Runs OCR for structural data (images only) + best VLM mode
- Checks model cache before attempting download
- Flags:
  - `--use-small` : force 1B model
  - `--mode ocr|janus|qwen` : force a specific mode
  - `--model-path <path>` : Janus local model dir
  - `--qwen-model-path <dir>` : Qwen GGUF model directory
  - `--video` : force video mode
  - `--video-fps 1.0` : frames per second for video extraction
  - `--video-max-frames 30` : max frames to process

### `scripts/qwen_analyze.py`
- Qwen2.5-VL-7B/72B GGUF model support (via llama-cpp-python)
- Image analysis with detailed description
- Video analysis: frame extraction (OpenCV) + frame-by-frame VLM + summary
- `--use-small` : prefer 7B over 72B
- `--model-path <dir>` : GGUF model directory
- `--video` : video mode
- Environment: `QWEN_MODEL_PATH`, `QWEN_GPU_LAYERS` (default: -1 = all), `QWEN_CTX_SIZE`

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
- Outputs GPU name, VRAM, RAM, Qwen GGUF status, and recommended mode

### `janus/` (cloned Janus-Pro source)
Used by janus_analyze.py for model inference.

## Qwen GGUF Setup (for Video Support)

1. Install llama-cpp-python:
```powershell
pip install llama-cpp-python
# For CUDA support:
$env:CMAKE_ARGS="-DGGML_CUDA=ON"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

2. Download Qwen2.5-VL GGUF model files from HuggingFace (e.g., bartowski):
   - Qwen2.5-VL-7B: main GGUF + mmproj GGUF
   - Qwen2.5-VL-72B: main GGUF + mmproj GGUF

3. Set env var or use `--qwen-model-path`:
```
$env:QWEN_MODEL_PATH = "F:\models\Qwen"
python scripts/analyze.py "video.mp4" --mode qwen
```

## Examples

User provides image path:
1. Run `python scripts/analyze.py "C:\path\to\image.png" 2>$null`
2. Parse JSON: hardware info, OCR text, structure grid, VLM description
3. Present merged summary to the user

User provides video path (Qwen GGUF set up):
1. Run `python scripts/analyze.py "C:\path\to\video.mp4" --mode qwen 2>$null`
2. Parse JSON: hardware info, video frames analyses, summary
3. Present video content summary to the user
