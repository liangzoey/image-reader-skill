#!/usr/bin/env python3
"""
Detect hardware configuration (GPU, VRAM, RAM) and recommend best analysis mode.
Outputs JSON with detection results and recommendation.

Usage:
    python auto_detect.py
"""
import sys, json, os, glob

def find_qwen_gguf(model_dir=None):
    """Check if Qwen GGUF model files exist in the given path or QWEN_MODEL_PATH."""
    search_dirs = []
    if model_dir and os.path.isdir(model_dir):
        search_dirs.append(model_dir)
    env_dir = os.environ.get("QWEN_MODEL_PATH")
    if env_dir and os.path.isdir(env_dir):
        search_dirs.append(env_dir)

    for d in search_dirs:
        models = glob.glob(os.path.join(d, "*.gguf"))
        mmprojs = [m for m in models if "mmproj" in os.path.basename(m).lower()]
        main_models = [m for m in models if "mmproj" not in os.path.basename(m).lower()]
        if main_models and mmprojs:
            sizes = []
            for m in main_models:
                name = os.path.basename(m).lower()
                if "72b" in name:
                    sizes.append("72b")
                elif "7b" in name:
                    sizes.append("7b")
            return {
                "found": True,
                "path": d,
                "models": [os.path.basename(m) for m in main_models],
                "sizes": sorted(set(sizes)),
            }
    return {"found": False}


def is_video_file(path):
    """Check if a file path is a supported video format."""
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
    ext = os.path.splitext(path)[1].lower()
    return ext in video_exts


def detect(model_path_hint=None):
    info = {
        "gpu": None,
        "vram_total_gb": None,
        "vram_free_gb": None,
        "ram_total_gb": None,
        "cuda_available": False,
        "recommended_mode": "ocr",
        "recommended_model": None,
        "recommendation": None,
        "video_support": False,
        "qwen_available": False,
        "qwen_info": None,
    }

    # Check CUDA / GPU
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_total_gb"] = round(props.total_memory / 1e9, 1)
            free, _ = torch.cuda.mem_get_info(0)
            info["vram_free_gb"] = round(free / 1e9, 1)
    except Exception:
        pass

    # Check system RAM
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            mem = ctypes.c_longlong()
            kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            # MEMORYSTATUSEX structure: dwLength, dwMemoryLoad, ullTotalPhys, ...
            info["ram_total_gb"] = round(mem / 1e9, 1)
        else:
            import psutil
            info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass

    # Check for Qwen GGUF model
    qwen = find_qwen_gguf(model_path_hint)
    info["qwen_available"] = qwen["found"]
    info["qwen_info"] = qwen

    # Check if input is a video file (hint from calling context)
    info["video_support"] = info["qwen_available"]  # Qwen supports video

    # Decide recommendation
    vram = info.get("vram_total_gb") or 0
    ram = info.get("ram_total_gb") or 0

    # Qwen gets priority if GGUF model is available (it's local and handles video)
    if info["qwen_available"]:
        sizes = qwen.get("sizes", [])
        if "72b" in sizes and vram >= 20:
            info["recommended_mode"] = "qwen"
            info["recommended_model"] = "72b"
            info["recommendation"] = f"Qwen2.5-VL-72B (GGUF, ~{vram}GB VRAM) — supports images + video"
        elif "7b" in sizes and vram >= 6:
            info["recommended_mode"] = "qwen"
            info["recommended_model"] = "7b"
            info["recommendation"] = f"Qwen2.5-VL-7B (GGUF, ~{vram}GB VRAM) — supports images + video"
        elif "7b" in sizes and ram >= 16:
            info["recommended_mode"] = "qwen"
            info["recommended_model"] = "7b_cpu"
            info["recommendation"] = f"Qwen2.5-VL-7B on CPU (GGUF, {ram}GB RAM) — supports images + video"
        else:
            info["qwen_available"] = False
            info["qwen_info"] = None
            info["recommended_mode"] = "ocr"
            info["recommendation"] = f"Qwen model found but insufficient hardware ({vram}GB VRAM, {ram}GB RAM)"
    elif info["cuda_available"] and vram >= 14:
        info["recommended_mode"] = "janus"
        info["recommended_model"] = "7b"
        info["recommendation"] = f"Janus-Pro-7B (FP16, ~{vram}GB VRAM available)"
    elif info["cuda_available"] and vram >= 8:
        info["recommended_mode"] = "janus"
        info["recommended_model"] = "7b_4bit"
        info["recommendation"] = "Janus-Pro-7B (4-bit, needs bitsandbytes)"
    elif info["cuda_available"] and vram >= 4:
        info["recommended_mode"] = "janus"
        info["recommended_model"] = "1b"
        info["recommendation"] = f"Janus-Pro-1B (~{vram}GB VRAM available)"
    elif ram >= 16:
        info["recommended_mode"] = "janus"
        info["recommended_model"] = "1b"
        info["recommendation"] = f"Janus-Pro-1B on CPU (no GPU, {ram}GB RAM)"
    else:
        info["recommended_mode"] = "ocr"
        info["recommendation"] = f"EasyOCR only (no GPU, {ram}GB RAM)"

    return info

if __name__ == "__main__":
    info = detect()
    print(json.dumps(info, ensure_ascii=False, indent=2))
