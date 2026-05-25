<div align="right">
  <a href="README.md">English</a> | <b>中文</b>
</div>

# Image Reader Skill for Claude Code / Claude Code 图片视频阅读技能

一个本地运行的图片/视频分析技能，让 [Claude Code](https://claude.ai/code) 能够通过 OCR（EasyOCR + OpenCV）、Janus-Pro 多模态 VLM 或 **Qwen2.5-VL GGUF 模型（支持视频分析）** 读取媒体内容——全部在本地运行，无需联网。

Claude Code 的模型无法直接处理图片或视频。这个技能通过在本地运行 Python 分析脚本，将媒体内容的结构化信息返回给 Claude。

---

## 功能特点

- **自动硬件检测** — 自动检测 GPU/显存/内存，检测 Qwen GGUF 模型是否存在
- **OCR 文字提取** — EasyOCR 支持中文简体 + 英文
- **图像结构分析** — OpenCV 4x4 网格分析（亮度、对比度、边缘密度、主色）
- **图像类型分类** — 启发式分类（截图、文档、照片、UI 等）
- **Janus-Pro 集成**（可选）— DeepSeek 多模态 VLM，用于深度语义理解
- **Qwen2.5-VL GGUF 支持**（可选）— 通过 llama.cpp 运行 Qwen2.5-VL（7B/14B/**32B**/72B），支持图片和**视频分析**
- **视频帧分析** — 从视频中提取关键帧，逐帧 VLM 分析并汇总
- **完全本地运行** — 无需云 API，数据不外传

---

## 硬件要求

| 模式 | 显存 | GPU | 内存 |
|------|------|-----|------|
| 仅 OCR | 任意 | 可选 | 任意 |
| Janus-Pro-1B (CPU) | 无 | 否 | >=16GB |
| Janus-Pro-1B (GPU) | >=4GB | 是 | 任意 |
| Janus-Pro-7B (FP16) | >=14GB | 是 | 任意 |
| Qwen2.5-VL-7B (GGUF) | >=6GB | 是 | 任意 |
| Qwen2.5-VL-32B (GGUF) | >=12GB | 是 | >=16GB |
| Qwen2.5-VL-14B (GGUF) | >=8GB | 是 | 任意 |
| Qwen2.5-VL-7B (GGUF) | >=6GB | 是 | 任意 |

每次运行时自动检测硬件并选择最佳模式。如果找到 Qwen GGUF 模型，优先使用（支持图片和视频）。

---

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision easyocr opencv-python-headless pillow
```

Janus-Pro（可选）：

```bash
pip install transformers timm attrdict
```

Qwen GGUF 视频分析（可选）：

```bash
pip install llama-cpp-python
```

### 2. 快速测试

```bash
python scripts/ocr_image.py "你的图片路径.png"
```

返回 JSON 格式的 OCR 文字、结构布局和图像分类信息。

### 3. 全功能分析（自动检测 + OCR + VLM）

```bash
python scripts/analyze.py "你的图片路径.png"
```

先自动检测硬件，然后自动选择最佳模式进行分析。

---

## 使用方式

### 基础 OCR + 结构分析

```bash
python scripts/ocr_image.py "image.png"
```

输出：检测到的文字、4x4 网格分析、尺寸、颜色、图片类型。

### 一键全分析（自动检测 + OCR + Janus/Qwen）

```bash
python scripts/analyze.py "image.png"
```

硬件检测 → OCR → 最佳 VLM（按照优先级自动选择已缓存的模型）

### 视频分析（需要 Qwen GGUF 模型）

```bash
# 自动检测 Qwen GGUF 模型
python scripts/analyze.py "video.mp4" --mode qwen

# 指定 GGUF 模型目录
python scripts/analyze.py "video.mp4" --mode qwen --qwen-model-path "F:/models/Qwen"

# 控制帧提取参数
python scripts/analyze.py "video.mp4" --mode qwen --video-fps 0.5 --video-max-frames 20
```

### 强制使用 Janus 模式

```bash
python scripts/analyze.py "image.png" --mode janus
python scripts/analyze.py "image.png" --mode janus --use-small
```

### 仅查看硬件信息

```bash
python scripts/auto_detect.py
```

---

## Qwen GGUF 配置（视频支持）

Qwen2.5-VL GGUF 格式模型同时支持图片和**视频分析**，这是推荐用于视频理解的模式。

### 第一步：安装 llama-cpp-python

```bash
# CPU only
pip install llama-cpp-python

# CUDA 加速（推荐）
$env:CMAKE_ARGS="-DGGML_CUDA=ON"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 第二步：下载 GGUF 模型文件

从 HuggingFace 下载（例如 [bartowski](https://huggingface.co/bartowski) 发布的版本）：

- **Qwen2.5-VL-7B**: 需要 2 个文件 — 主模型 GGUF + mmproj GGUF（~5GB）
- **Qwen2.5-VL-7B/14B/32B/72B**: 需要 2 个文件 — 主模型 GGUF + mmproj GGUF

放在同一目录下，如 `F:/models/Qwen/`。

### 第三步：设置环境变量

```bash
# 可选：设置默认模型路径
set QWEN_MODEL_PATH=F:/models/Qwen

# 可选：配置 GPU 层数（默认 -1 = 全部层 GPU 加速）
set QWEN_GPU_LAYERS=-1

# 可选：配置上下文大小（默认 8192）
set QWEN_CTX_SIZE=8192
```

### 第四步：运行视频分析

```bash
python scripts/analyze.py "视频.mp4" --mode qwen
```

---

## 安装为 Claude Code 技能

1. 将 `image-reader` 文件夹放入 `~/.claude/skills/` 目录
2. 重启 Claude Code
3. 当你在对话中提供图片或视频路径时，Claude 会自动调用此技能

或者使用打包好的技能文件通过 Claude Code 技能管理功能安装。

---

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `scripts/analyze.py` | 主入口 — 自动检测硬件，运行 OCR + 最佳 VLM（Janus/Qwen） |
| `scripts/ocr_image.py` | EasyOCR 文字提取 + OpenCV 结构分析 |
| `scripts/janus_analyze.py` | Janus-Pro-7B/1B 深度图片理解 |
| `scripts/qwen_analyze.py` | Qwen2.5-VL GGUF（7B/14B/32B/72B）图片 + 视频分析 |
| `scripts/auto_detect.py` | 硬件检测 + Qwen GGUF 模型发现 |

---

## Janus-Pro 配置（可选）

Janus-Pro 模型会在首次使用时自动从 HuggingFace 下载：

- **Janus-Pro-7B**: ~15GB（[deepseek-ai/Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)）
- **Janus-Pro-1B**: ~2GB（[deepseek-ai/Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)）

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/Janus-Pro-7B')"
```

或使用本地模型目录：

```bash
python scripts/analyze.py "image.png" --model-path "F:/ComfyUIClassic/ComfyUIClassic/models/Janus-Pro"
```

---

## 输出格式

### OCR + 结构输出

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

### 图片全分析输出

```json
{
  "file": "path/to/image.png",
  "hardware": { "gpu": "NVIDIA GeForce RTX 4090", "vram_total_gb": 24.0, ... },
  "mode_used": "qwen",
  "ocr_text": [...],
  "structure": {...},
  "description": "这是一张包含数据仪表盘的截图...",
  "total_time_s": 12.3
}
```

### 视频分析输出

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
    { "frame_index": 0, "analysis": "一个人走进房间...", "time_s": 1.2 },
    { "frame_index": 60, "analysis": "这个人坐在桌前...", "time_s": 1.1 }
  ],
  "total_time_s": 18.5
}
```

---

## 技术细节

- **OCR 引擎**: EasyOCR（CRNN + CTC 解码器），支持 ["ch_sim", "en"] 语言
- **预处理**: 3 倍放大 + 4 种预处理（原图、灰度、Otsu、反转 Otsu）
- **网格分析**: 4x4 区域分解，每单元格亮度、对比度、边缘密度、主色
- **图像分类**: 基于边缘密度、亮度方差和对比度的启发式算法
- **VLM 1**: Janus-Pro（SigLIP 视觉编码器 + Llama 语言模型），直接 `from_pretrained`
- **VLM 2**: Qwen2.5-VL 通过 llama.cpp GGUF，支持图片和视频
- **视频流程**: OpenCV 帧提取 → 逐帧 VLM 分析 → 帧级结果 + 汇总

---

## 许可

MIT
