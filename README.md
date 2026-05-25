# Image Reader Skill for Claude Code / Claude Code 图片阅读技能

一个本地运行的图片分析技能，让 [Claude Code](https://claude.ai/code) 能够通过 OCR（EasyOCR + OpenCV）和可选的 Janus-Pro 多模态 VLM 读取图片内容——全部在本地运行，无需联网。

Claude Code 的模型（如 deepseek-v4-flash、Claude Opus 等）无法直接处理图片。这个技能通过在本地运行 Python 分析脚本，将图片的结构化信息和文本内容返回给 Claude，填补了这一空白。

---

## 功能特点 / Features

- **自动硬件检测** — 自动检测 GPU/VRAM/内存，选择最优分析模式
- **OCR 文字提取** — EasyOCR 支持中文简体 + 英文
- **图像结构分析** — OpenCV 4x4 网格分析（亮度、对比度、边缘密度、主色）
- **图像类型分类** — 启发式分类（截图、文档、照片、UI 等）
- **Janus-Pro 集成**（可选）— DeepSeek 多模态 VLM，用于深度语义理解
- **完全本地运行** — 无需云 API，数据不外传

---

## 硬件要求 / Hardware Requirements

| 模式 / Mode | 显存 / VRAM | GPU | 内存 / RAM |
|-------------|-------------|-----|-----------|
| 仅 OCR / OCR only | 任意 | 可选 | 任意 |
| Janus-Pro-1B (CPU) | 无 | 否 | >=16GB |
| Janus-Pro-1B (GPU) | >=4GB | 是 | 任意 |
| Janus-Pro-7B (4-bit) | >=8GB | 是 | 任意 |
| Janus-Pro-7B (FP16) | >=14GB | 是 | 任意 |

每次运行时自动检测硬件并选择最佳模式。

---

## 快速开始 / Quick Start

### 1. 安装依赖 / Install Dependencies

```bash
pip install torch torchvision easyocr opencv-python-headless pillow
```

如果需要 Janus-Pro（可选，用于深度理解图片内容）：

```bash
pip install transformers timm attrdict
```

### 2. 快速测试 / Quick Test

```bash
python scripts/ocr_image.py "你的图片路径.png"
```

返回 JSON 格式的 OCR 文字、结构布局和图像分类信息。

### 3. 全功能分析 / Full Analysis (OCR + 可选 Janus)

```bash
python scripts/analyze.py "你的图片路径.png"
```

先自动检测硬件，然后自动选择最佳模式进行分析。

---

## 使用方式 / Usage

### 基础 OCR + 结构分析 / Basic OCR + Structural Analysis

```bash
python scripts/ocr_image.py "image.png"
```

输出：检测到的文字、4x4 网格分析、尺寸、颜色、图片类型。

### 一键全分析 / All-in-One (Auto-Detect + OCR + Janus)

```bash
python scripts/analyze.py "image.png"
```

硬件检测 → OCR（始终运行）→ Janus-Pro（如果显存/内存足够且模型已缓存）

### 强制使用 Janus 模式 / Force Janus Mode

```bash
# 使用 7B 模型（需要 ~14GB 显存）
python scripts/analyze.py "image.png" --mode janus

# 使用 1B 模型（需要 ~4GB 显存或 ~16GB 内存）
python scripts/analyze.py "image.png" --mode janus --use-small
```

### 仅查看硬件信息 / Hardware Info Only

```bash
python scripts/auto_detect.py
```

---

## 安装为 Claude Code 技能 / Installing as a Claude Code Skill

1. 将 `image-reader` 文件夹放入 `~/.claude/skills/` 目录
2. 重启 Claude Code
3. 当你在对话中提供图片路径时，Claude 会自动调用此技能

或者使用打包好的技能文件：

```bash
# .skill 文件可以通过 Claude Code 的技能管理功能安装
```

---

## 脚本说明 / Scripts Overview

| 脚本 / Script | 功能 / Purpose |
|---------------|----------------|
| `scripts/analyze.py` | 主入口 — 自动检测硬件，运行 OCR + 可选 Janus |
| `scripts/ocr_image.py` | EasyOCR 文字提取 + OpenCV 结构分析 |
| `scripts/janus_analyze.py` | Janus-Pro-7B/1B 深度图片理解 |
| `scripts/auto_detect.py` | 独立的硬件检测（GPU、显存、内存） |

---

## Janus-Pro 配置（可选）/ Janus-Pro Setup (Optional)

Janus-Pro 模型会在首次使用时自动从 HuggingFace 下载：

- **Janus-Pro-7B**: ~15GB 下载 ([deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B))
- **Janus-Pro-1B**: ~2GB 下载 ([deepseek-ai/Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B))

预下载命令：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/Janus-Pro-7B')"
```

或者使用本地模型目录：

```bash
python scripts/analyze.py "image.png" --model-path "F:/ComfyUIClassic/ComfyUIClassic/models/Janus-Pro"
```

---

## 输出格式 / Output Format

### OCR + 结构输出 / OCR + Structure Output

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

### 全分析输出 / Full Analysis Output

```json
{
  "file": "path/to/image.png",
  "hardware": { "gpu": "NVIDIA GeForce RTX 4090", "vram_total_gb": 24.0, ... },
  "mode_used": "janus",
  "ocr_text": [...],
  "structure": {...},
  "description": "这是一张截图，显示...",
  "total_time_s": 12.3
}
```

---

## 技术细节 / Technical Details

- **OCR 引擎**: EasyOCR（CRNN + CTC 解码器），支持 ["ch_sim", "en"] 语言
- **预处理**: 3 倍放大 + 4 种预处理（原图、灰度、Otsu、反转 Otsu）
- **网格分析**: 4x4 区域分解，每单元格亮度、对比度、边缘密度、主色
- **图像分类**: 基于边缘密度、亮度方差和对比度的启发式算法
- **VLM**: Janus-Pro（SigLIP 视觉编码器 + Llama 语言模型），直接通过 `from_pretrained` 导入

---

## 许可 / License

MIT
