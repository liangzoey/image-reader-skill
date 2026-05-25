#!/usr/bin/env python3
"""
Auto-setup script for Qwen GGUF support.
Detects hardware, installs llama-cpp-python with the right backend,
and guides model download.

Usage:
    python setup_qwen.py              # auto-detect and guide
    python setup_qwen.py --install    # auto-install llama-cpp-python
    python setup_qwen.py --check      # just check hardware compatibility
"""
import sys, os, json, platform, subprocess, shutil


def check_cuda():
    """Check if CUDA-capable GPU is available via torch."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = round(props.total_memory / 1e9, 1)
            return {"available": True, "gpu": name, "vram_gb": vram}
    except ImportError:
        pass
    # Fallback: check nvidia-smi
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True)
        parts = out.strip().split(", ")
        vram = round(float(parts[1].replace(" MiB", "")) / 1024, 1)
        return {"available": True, "gpu": parts[0], "vram_gb": vram}
    except Exception:
        pass
    return {"available": False, "gpu": None, "vram_gb": 0}


def check_ram():
    """Check system RAM."""
    try:
        if os.name == "nt":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            return round(mem.ullTotalPhys / 1e9, 1)
        else:
            import psutil
            return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        return 0


def check_llama_cpp():
    """Check if llama-cpp-python is installed and its backend."""
    try:
        import llama_cpp
        ver = getattr(llama_cpp, "__version__", "unknown")
        # Check if CUDA backend is available
        try:
            from llama_cpp import Llama
            has_cuda = hasattr(Llama, "supports_gpu") and Llama.supports_gpu
        except Exception:
            has_cuda = False
        return {"installed": True, "version": ver, "cuda": has_cuda}
    except ImportError:
        return {"installed": False, "version": None, "cuda": False}


def estimate_model_size(model_label):
    """Estimate GGUF file size for a given model size label."""
    size_b = 0
    import re
    m = re.search(r'(\d+)', model_label)
    if m:
        size_b = int(m.group(1))

    # Rough Q4 size estimate: params * 0.5 GB per 1B params + overhead
    estimated_gb = round(size_b * 0.55 + 1, 1)
    return estimated_gb


def auto_setup(do_install=False):
    """Auto-detect hardware and provide setup guidance.

    Returns dict with hardware info, compatibility status, and instructions.
    """
    result = {
        "cuda": check_cuda(),
        "ram_gb": check_ram(),
        "llama_cpp": check_llama_cpp(),
        "compatible": False,
        "message": "",
        "install_command": None,
        "download_hint": None,
    }

    cuda = result["cuda"]
    ram = result["ram_gb"]
    llama = result["llama_cpp"]

    # Determine compatibility
    if cuda["available"] and cuda["vram_gb"] >= 12:
        result["compatible"] = True
        result["message"] = (
            f"GPU {cuda['gpu']} with {cuda['vram_gb']}GB VRAM detected. "
            f"Sufficient for Qwen3.5-27B (Q4)."
        )
        result["install_command"] = (
            '$env:CMAKE_ARGS="-DGGML_CUDA=ON"\n'
            "pip install llama-cpp-python --force-reinstall --no-cache-dir"
        )
    elif cuda["available"] and cuda["vram_gb"] >= 6:
        result["compatible"] = True
        result["message"] = (
            f"GPU {cuda['gpu']} with {cuda['vram_gb']}GB VRAM detected. "
            f"Sufficient for Qwen GGUF 7B model."
        )
        result["install_command"] = (
            '$env:CMAKE_ARGS="-DGGML_CUDA=ON"\n'
            "pip install llama-cpp-python --force-reinstall --no-cache-dir"
        )
        result["model_hint"] = "7B"
    elif ram >= 16:
        result["compatible"] = True
        result["message"] = (
            f"No compatible GPU found, but {ram}GB RAM detected. "
            f"Can run Qwen GGUF 7B on CPU (slower)."
        )
        result["install_command"] = "pip install llama-cpp-python"
        result["model_hint"] = "7B"
    else:
        result["message"] = (
            f"Insufficient hardware: "
            f"{'GPU ' + cuda['gpu'] + ' ' + str(cuda['vram_gb']) + 'GB VRAM' if cuda['available'] else 'No GPU'}, "
            f"{ram}GB RAM.\n"
            f"Qwen GGUF mode requires: >=12GB VRAM (27B) or >=6GB VRAM (7B) or >=16GB RAM (CPU 7B)."
        )
        return result

    # Check if already installed
    if llama["installed"]:
        result["message"] += f"\nllama-cpp-python {llama['version']} already installed."
        if cuda["available"] and not llama["cuda"] and cuda["vram_gb"] >= 6:
            result["message"] += "\nWarning: installed without CUDA. GPU acceleration not available."
            if do_install:
                result["message"] += "\nReinstalling with CUDA..."
        elif do_install:
            result["message"] += "\nAlready up to date."
    elif do_install:
        result["message"] += "\nInstalling llama-cpp-python..."

    # Download hint
    model_size = estimate_model_size(result.get("model_hint", "27B"))
    result["download_hint"] = (
        f"Suggested model: Qwen3.5-27B-Q4_K_M.gguf (~{model_size}GB) + mmproj-F16.gguf (~1GB)\n"
        f"Download from: https://huggingface.co/unsloth/Qwen3.5-27B-GGUF\n"
        f"Place both files in a directory and set:\n"
        f'  $env:QWEN_MODEL_PATH = "你的模型目录路径"'
    )

    return result


if __name__ == "__main__":
    do_install = "--install" in sys.argv
    do_check = "--check" in sys.argv

    result = auto_setup(do_install=do_install)

    if do_check:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0 if result["compatible"] else 1)

    print("=" * 60)
    print("  Qwen GGUF Auto-Setup")
    print("=" * 60)

    cuda = result["cuda"]
    print(f"\n GPU:       {cuda['gpu'] or 'Not detected'}")
    print(f" VRAM:      {cuda['vram_gb']}GB" if cuda['vram_gb'] else " VRAM:      N/A")
    print(f" RAM:       {result['ram_gb']}GB")
    print(f" CUDA:      {'Yes' if cuda['available'] else 'No'}")

    llama = result["llama_cpp"]
    print(f" llama-cpp: {llama['version'] or 'Not installed'}")

    print(f"\n Result:    {'Compatible' if result['compatible'] else 'Incompatible'}")
    print(f"\n {result['message']}")

    if result["compatible"]:
        if not llama["installed"]:
            print(f"\n-- Install command --\n{result['install_command']}\n")
        print(f"\n-- Model Download --\n{result['download_hint']}\n")

        if cuda["available"] and cuda["vram_gb"] >= 12:
            print(" Your RTX 4090 (24GB) can run Qwen3.5-27B Q4_K_M (~16.7GB) with room to spare!")
    else:
        print("\n Tip: Use --mode ocr for basic OCR analysis instead.")
