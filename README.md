# Image Reader Skill for Claude Code

A local image analysis skill for [Claude Code](https://claude.ai/code) that enables image reading capabilities via OCR (EasyOCR + OpenCV) and optionally Janus-Pro multimodal VLM — all running locally on your machine.

Claude Code models (like deepseek-v4-flash, Claude Opus, etc.) cannot directly process images. This skill bridges that gap by running Python-based analysis locally and returning structured results.

## Features

- **Automatic Hardware Detection** — Detects GPU/VRAM/RAM and picks the best available analysis mode
- **OCR Text Extraction** — EasyOCR with Chinese (Simplified) + English support
- **Structural Image Analysis** — OpenCV-based 4x4 grid analysis (brightness, contrast, edge density, dominant colors)
- **Image Type Classification** — Heuristic classification (screenshot, document, photo, UI, etc.)
- **Janus-Pro Integration** (optional) — DeepSeek's multimodal VLM for deep semantic image understanding
- **No Cloud API Needed** — Everything runs locally, no data leaves your machine

## Hardware Requirements

| Mode | VRAM | GPU | RAM |
|------|------|-----|-----|
| OCR only | Any | Optional | Any |
| Janus-Pro-1B (CPU) | N/A | No | >=16GB |
| Janus-Pro-1B (GPU) | >=4GB | Yes | Any |
| Janus-Pro-7B (4-bit) | >=8GB | Yes | Any |
| Janus-Pro-7B (FP16) | >=14GB | Yes | Any |

Auto-detection selects the best mode on each run.

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision easyocr opencv-python-headless pillow
```

For Janus-Pro (optional, for deep image understanding):

```bash
pip install transformers timm attrdict
```

### 2. Run a Quick Test

```bash
python scripts/ocr_image.py "path/to/your/image.png"
```

This returns JSON with OCR text, structural layout, and image classification.

### 3. Run Full Analysis (OCR + optional Janus)

```bash
python scripts/analyze.py "path/to/your/image.png"
```

The hardware auto-detection runs first, and the script selects the best mode automatically.

## Usage

### Basic OCR + Structural Analysis

```bash
python scripts/ocr_image.py "image.png"
```

Output: JSON with detected text, 4x4 grid analysis, dimensions, colors, image type.

### All-in-One (Auto-Detect + OCR + Janus)

```bash
python scripts/analyze.py "image.png"
```

Hardware detection → OCR (always runs) → Janus-Pro (if VRAM/RAM sufficient + model cached)

### Force Janus Mode

```bash
# Use the 7B model (requires ~14GB VRAM)
python scripts/analyze.py "image.png" --mode janus

# Use the 1B model (requires ~4GB VRAM or ~16GB RAM)
python scripts/analyze.py "image.png" --mode janus --use-small
```

### Hardware Info Only

```bash
python scripts/auto_detect.py
```

## Installing as a Claude Code Skill

1. Place the `image-reader` folder in your `~/.claude/skills/` directory
2. Restart Claude Code
3. When you provide an image path, Claude will automatically invoke this skill

Or use the packaged skill file:

```bash
# The .skill file can be installed via Claude Code's skill management
```

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `scripts/analyze.py` | Main entry point — auto-detects hardware, runs OCR + optional Janus |
| `scripts/ocr_image.py` | EasyOCR text extraction + OpenCV structural analysis |
| `scripts/janus_analyze.py` | Janus-Pro-7B/1B deep image understanding |
| `scripts/auto_detect.py` | Standalone hardware detection (GPU, VRAM, RAM) |

## Janus-Pro Setup (Optional)

The Janus-Pro models are downloaded automatically from HuggingFace on first use:

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

### Full Analysis Output

```json
{
  "file": "path/to/image.png",
  "hardware": { "gpu": "NVIDIA GeForce RTX 4090", "vram_total_gb": 24.0, ... },
  "mode_used": "janus",
  "ocr_text": [...],
  "structure": {...},
  "description": "This is a screenshot showing...",
  "total_time_s": 12.3
}
```

## Technical Details

- **OCR Engine**: EasyOCR (CRNN + CTC decoder) with ["ch_sim", "en"] languages
- **Preprocessing**: 3x upscale + 4 variants (original, grayscale, Otsu, inverted Otsu)
- **Grid Analysis**: 4x4 region breakdown with per-cell brightness, contrast, edge density, dominant color
- **Image Classification**: Heuristics based on edge density, brightness variance, and contrast
- **VLM**: Janus-Pro (SigLIP vision encoder + Llama language model) via direct `from_pretrained` import

## License

MIT
