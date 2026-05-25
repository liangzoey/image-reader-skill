<div align="right">
  <b>English</b> | <a href="README.zh.md">中文</a>
</div>

# Image Reader Skill for Claude Code

A local image/video analysis skill for [Claude Code](https://claude.ai/code) that enables media understanding via OCR (EasyOCR + OpenCV), Janus-Pro multimodal VLM, or **Qwen2.5-VL GGUF models (with video support)** — all running locally on your machine.

Claude Code models cannot directly process images or video. This skill bridges that gap by running Python-based analysis locally and returning structured results.

---

## Features

- **Automatic Hardware Detection** — Detects GPU/VRAM/RAM and Qwen GGUF model availability
- **OCR Text Extraction** — EasyOCR with Chinese (Simplified) + English support
- **Structural Image Analysis** — OpenCV-based 4x4 grid analysis (brightness, contrast, edge density, dominant colors)
- **Image Type Classification** — Heuristic classification (screenshot, document, photo, UI, etc.)
- **Janus-Pro Integration** (optional) — DeepSeek's multimodal VLM for deep semantic image understanding
- **Qwen2.5-VL GGUF Support** (optional) — Run Qwen2.5-VL-7B/72B via llama.cpp for both image and **video analysis**
- **Video Frame Analysis** — Extracts frames from video files and analyzes them frame-by-frame with VLM
- **No Cloud API Needed** — Everything runs locally, no data leaves your machine

---

## Hardware Requirements

| Mode | VRAM | GPU | RAM |
|------|------|-----|-----|
| OCR only | Any | Optional | Any |
| Janus-Pro-1B (CPU) | N/A | No | >=16GB |
| Janus-Pro-1B (GPU) | >=4GB | Yes | Any |
| Janus-Pro-7B (4-bit) | >=8GB | Yes | Any |
| Janus-Pro-7B (FP16) | >=14GB | Yes | Any |
| Qwen2.5-VL-7B (GGUF) | >=6GB | Yes | Any |
| Qwen2.5-VL-72B (GGUF) | >=20GB | Yes | >=16GB |

Auto-detection selects the best mode on each run. If a Qwen GGUF model is found, it takes priority (supports both images and video).

---

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision easyocr opencv-python-headless pillow
```

For Janus-Pro (optional):

```bash
pip install transformers timm attrdict
```

For Qwen GGUF video analysis (optional):

```bash
pip install llama-cpp-python
```

### 2. Quick Test

```bash
python scripts/ocr_image.py "path/to/your/image.png"
```

Returns JSON with OCR text, structural layout, and image classification.

### 3. Full Analysis (Auto-Detect + OCR + VLM)

```bash
python scripts/analyze.py "path/to/your/image.png"
```

Hardware auto-detection runs first, selecting the best mode automatically.

---

## Usage

### Basic OCR + Structural Analysis

```bash
python scripts/ocr_image.py "image.png"
```

Output: JSON with detected text, 4x4 grid analysis, dimensions, colors, image type.

### All-in-One (Auto-Detect + OCR + Janus/Qwen)

```bash
python scripts/analyze.py "image.png"
```

Hardware detection → OCR (always runs) → Best VLM (cached model with highest priority)

### Video Analysis (requires Qwen GGUF model)

```bash
# Auto-detect Qwen GGUF model
python scripts/analyze.py "video.mp4" --mode qwen

# Specify GGUF model directory
python scripts/analyze.py "video.mp4" --mode qwen --qwen-model-path "F:/models/Qwen"

# Control frame extraction
python scripts/analyze.py "video.mp4" --mode qwen --video-fps 0.5 --video-max-frames 20
```

### Force Janus Mode

```bash
python scripts/analyze.py "image.png" --mode janus
python scripts/analyze.py "image.png" --mode janus --use-small
```

### Hardware Info Only

```bash
python scripts/auto_detect.py
```

---

## Qwen GGUF Setup (for Video Support)

Qwen2.5-VL models in GGUF format support both image and **video** analysis via llama.cpp. This is the recommended mode if you need video understanding.

### Step 1: Install llama-cpp-python

```bash
# CPU only
pip install llama-cpp-python

# With CUDA support (recommended for GPU acceleration)
set CMAKE_ARGS=-DGGML_CUDA=ON
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Step 2: Download GGUF Model Files

Download from HuggingFace (e.g., [bartowski](https://huggingface.co/bartowski)):

- **Qwen2.5-VL-7B**: Needs 2 files — main model GGUF + mmproj GGUF (~5GB total)
- **Qwen2.5-VL-72B**: Needs 2 files — main model GGUF + mmproj GGUF (~45GB total)

Place them in a directory like `F:/models/Qwen/`.

### Step 3: Set Environment Variable

```bash
# Optional: set default model path
set QWEN_MODEL_PATH=F:/models/Qwen

# Optional: configure GPU layers (default: all layers on GPU)
set QWEN_GPU_LAYERS=-1

# Optional: configure context size (default: 8192)
set QWEN_CTX_SIZE=8192
```

### Step 4: Run Video Analysis

```bash
python scripts/analyze.py "video.mp4" --mode qwen
```

---

## Installing as a Claude Code Skill

1. Place the `image-reader` folder in your `~/.claude/skills/` directory
2. Restart Claude Code
3. When you provide an image or video path, Claude will automatically invoke this skill

Or use the packaged skill file:

```bash
# The .skill file can be installed via Claude Code's skill management
```

---

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `scripts/analyze.py` | Main entry point — auto-detects hardware, runs OCR + best VLM (Janus/Qwen) |
| `scripts/ocr_image.py` | EasyOCR text extraction + OpenCV structural analysis |
| `scripts/janus_analyze.py` | Janus-Pro-7B/1B deep image understanding |
| `scripts/qwen_analyze.py` | Qwen2.5-VL-7B/72B GGUF image + video analysis |
| `scripts/auto_detect.py` | Hardware detection + Qwen GGUF model discovery |

---

## Janus-Pro Setup (Optional)

Janus-Pro models are downloaded automatically from HuggingFace on first use:

- **Janus-Pro-7B**: ~15GB download ([deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B))
- **Janus-Pro-1B**: ~2GB download ([deepseek-ai/Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B))

To pre-download:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/Janus-Pro-7B')"
```

Or use a local model directory:

```bash
python scripts/analyze.py "image.png" --model-path "F:/ComfyUIClassic/ComfyUIClassic/models/Janus-Pro"
```

---

## Output Format

### OCR + Structure Output

```json
{
  "file": "path/to/image.png",
  "text_found": [{"text": "Hello World", "confidence": 0.95, "method": "otsu"}],
  "structure": {
    "dimensions": "1920x1080",
    "aspect_ratio": 1.778,
    "brightness": 128.5,
    "contrast": 45.2,
    "edge_density_percent": 5.1,
    "likely_type": "screenshot/UI",
    "grid_4x4": [...]
  }
}
```

### Full Image Analysis Output

```json
{
  "file": "path/to/image.png",
  "hardware": { "gpu": "NVIDIA GeForce RTX 4090", "vram_total_gb": 24.0, ... },
  "mode_used": "janus",
  "ocr_text": [...],
  "structure": {...},
  "description": "This is a screenshot showing a dashboard with various charts...",
  "total_time_s": 12.3
}
```

### Video Analysis Output

```json
{
  "file": "path/to/video.mp4",
  "mode_used": "qwen",
  "hardware": { "gpu": "NVIDIA GeForce RTX 4090", "qwen_available": true, ... },
  "video_info": {
    "video_fps": 30.0,
    "total_frames": 900,
    "extracted_frames": 15,
    "sample_interval_frames": 60
  },
  "description": "[Video analysis: 15 frames from 900 total frames]\n\nFrame-by-frame descriptions:\n...",
  "video_analysis": [
    { "frame_index": 0, "analysis": "A person walking into a room...", "time_s": 1.2 },
    { "frame_index": 60, "analysis": "The person sits down at a desk...", "time_s": 1.1 }
  ],
  "total_time_s": 18.5
}
```

---

## Technical Details

- **OCR Engine**: EasyOCR (CRNN + CTC decoder) with ["ch_sim", "en"] languages
- **Preprocessing**: 3x upscale + 4 variants (original, grayscale, Otsu, inverted Otsu)
- **Grid Analysis**: 4x4 region breakdown with per-cell brightness, contrast, edge density, dominant color
- **Image Classification**: Heuristics based on edge density, brightness variance, and contrast
- **VLM**: Janus-Pro (SigLIP vision encoder + Llama language model) via direct `from_pretrained`
- **Qwen GGUF**: Qwen2.5-VL via llama.cpp with multimodal projection, supports both images and video
- **Video Pipeline**: OpenCV frame extraction → per-frame VLM analysis → frame-by-frame results with summary

---

## License

MIT
