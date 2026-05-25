#!/usr/bin/env python3
"""
Detect hardware configuration (GPU, VRAM, RAM) and recommend best analysis mode.
Outputs JSON with detection results and recommendation.

Usage:
    python auto_detect.py
"""
import sys, json, os, subprocess

def detect():
    info = {
        "gpu": None,
        "vram_total_gb": None,
        "vram_free_gb": None,
        "ram_total_gb": None,
        "cuda_available": False,
        "recommended_mode": "ocr",
        "recommended_model": None,
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

    # Decide recommendation
    vram = info.get("vram_total_gb") or 0
    ram = info.get("ram_total_gb") or 0

    if info["cuda_available"]:
        if vram >= 14:
            info["recommended_mode"] = "janus"
            info["recommended_model"] = "7b"
            info["recommendation"] = f"Janus-Pro-7B (FP16, ~{vram}GB VRAM available)"
        elif vram >= 8:
            info["recommended_mode"] = "janus"
            info["recommended_model"] = "7b_4bit"
            info["recommendation"] = "Janus-Pro-7B (4-bit, needs bitsandbytes)"
        elif vram >= 4:
            info["recommended_mode"] = "janus"
            info["recommended_model"] = "1b"
            info["recommendation"] = f"Janus-Pro-1B (~{vram}GB VRAM available)"
        else:
            info["recommended_mode"] = "ocr"
            info["recommendation"] = f"EasyOCR only (insufficient VRAM: {vram}GB)"
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
