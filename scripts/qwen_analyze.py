#!/usr/bin/env python3
"""
Analyze images and videos using Qwen2.5-VL models in GGUF format.
Supports image description, OCR, and video frame-by-frame analysis.

Usage:
    python qwen_analyze.py <image_path> [prompt]
    python qwen_analyze.py <video_path> [prompt] --video
    python qwen_analyze.py <path> --model-path <gguf_dir> [--use-small]

Environment:
    QWEN_MODEL_PATH     Directory containing Qwen GGUF files
    QWEN_GPU_LAYERS     Number of layers to offload to GPU (-1 = all, default: -1)
    QWEN_CTX_SIZE       Context size (default: 8192)
"""
import sys, os, json, time, logging, warnings, tempfile
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# Minimum VRAM thresholds for different model size ranges (GB)
MODEL_VRAM_THRESHOLDS = [
    (70, 20.0),   # 70B+ range (72B, 70B)
    (30, 12.0),   # 30B+ range (32B, 35B, 27B)
    (13, 8.0),    # 13B+ range (14B)
    (6, 6.0),     # 6B+ range (7B, 8B)
    (0, 4.0),     # anything smaller
]


def parse_model_size(label):
    """Extract approximate parameter count from a model label string."""
    import re
    sizes = re.findall(r'(\d+)\s*[bB]', label)
    if sizes:
        return int(sizes[0])
    return 0


def find_qwen_gguf_files(model_dir):
    """Find all GGUF + mmproj pairs in a directory.

    Returns sorted list of (model_path, mmproj_path, size_label).
    """
    import glob
    all_gguf = glob.glob(os.path.join(model_dir, "*.gguf"))
    all_gguf.extend(glob.glob(os.path.join(model_dir, "**", "*.gguf"), recursive=True))
    all_gguf = sorted(set(all_gguf))

    mmprojs = [m for m in all_gguf if "mmproj" in os.path.basename(m).lower()]
    models = [m for m in all_gguf if "mmproj" not in os.path.basename(m).lower()]

    # If no mmproj found, VLM won't work
    if not mmprojs:
        return []

    results = []
    for model_path in models:
        label = os.path.basename(model_path)
        matched_mmproj = None

        # Strategy 1: match by shared size number (e.g., "27B" in model name + mmproj name)
        model_base = os.path.splitext(os.path.basename(model_path))[0].lower()
        for mp in mmprojs:
            mp_base = os.path.splitext(os.path.basename(mp))[0].lower()
            model_nums = set(re.findall(r'(\d+)[bB]', model_base))
            mp_nums = set(re.findall(r'(\d+)[bB]', mp_base))
            if model_nums & mp_nums:
                matched_mmproj = mp
                break

        # Strategy 2: fallback — pick first mmproj (handles models like Qwen3.5-27B
        # where mmproj is named mmproj-F16.gguf with no model size in name)
        if matched_mmproj is None and mmprojs:
            matched_mmproj = mmprojs[0]

        if matched_mmproj:
            size_b = parse_model_size(label)
            results.append((model_path, matched_mmproj, label, size_b))

    # Sort by model size descending (largest model first)
    results.sort(key=lambda x: x[3], reverse=True)
    return results

    # Sort by model size descending (largest model first)
    results.sort(key=lambda x: x[3], reverse=True)
    return results


def pick_best_model(model_dir=None, prefer_small=False, vram_gb=0):
    """Find Qwen GGUF model files, picking the best fit for hardware.

    Returns (model_path, mmproj_path, model_label) or (None, None, None).
    """
    search_dirs = []
    if model_dir and os.path.isdir(model_dir):
        search_dirs.append(model_dir)
    env_dir = os.environ.get("QWEN_MODEL_PATH")
    if env_dir and os.path.isdir(env_dir):
        search_dirs.append(env_dir)

    all_models = []
    for d in search_dirs:
        all_models.extend(find_qwen_gguf_files(d))

    if not all_models:
        return None, None, None

    if prefer_small:
        # Pick smallest model
        chosen = all_models[-1]
    else:
        # Pick largest model that fits in VRAM
        chosen = None
        for m in all_models:
            size_b = m[3]
            min_vram = 4.0
            for size_threshold, vram_req in MODEL_VRAM_THRESHOLDS:
                if size_b >= size_threshold:
                    min_vram = vram_req
                    break
            if vram_gb >= min_vram or chosen is None:
                chosen = m
                if vram_gb >= min_vram:
                    break  # largest fitting model

        if chosen is None:
            chosen = all_models[0]

    return chosen[0], chosen[1], chosen[2]


def extract_video_frames(video_path, fps=1.0, max_frames=30):
    """Extract frames from a video file at given fps.

    Returns list of (frame_index, pil_image).
    """
    try:
        import cv2
        from PIL import Image
    except ImportError:
        return None, "OpenCV (cv2) required for video processing. pip install opencv-python-headless"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Could not open video: {video_path}"

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(video_fps / fps))

    frames = []
    frame_idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append((frame_idx, pil_img))
        frame_idx += 1

    cap.release()

    info = {
        "video_fps": round(video_fps, 1),
        "total_frames": total_frames,
        "extracted_frames": len(frames),
        "sample_interval_frames": sample_interval,
    }
    return frames, info


def analyze_media(media_path, prompt=None, model_dir=None, use_small=False, is_video=False,
                  video_fps=1.0, video_max_frames=30):
    """Analyze an image or video using a Qwen2.5-VL GGUF model.

    Args:
        media_path: Path to image or video file
        prompt: Optional text prompt
        model_dir: Directory containing Qwen GGUF files
        use_small: Prefer smallest available model (e.g., 7B over 32B/72B)
        is_video: Treat input as video
        video_fps: Frames per second to extract for video
        video_max_frames: Maximum frames to process

    Returns:
        dict with analysis results
    """
    result = {
        "file": media_path,
        "model": None,
        "error": None,
        "analysis": None,
        "video_info": None,
        "frame_analyses": None,
    }

    # Find model with VRAM-aware selection
    try:
        import torch
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0
    except Exception:
        vram_gb = 0

    model_path, mmproj_path, model_label = pick_best_model(model_dir, prefer_small=use_small, vram_gb=vram_gb)
    if not model_path:
        # Auto-download model
        print("  [qwen] No GGUF model found. Auto-downloading Qwen3.5-27B...", file=sys.stderr)
        model_dir = auto_download_model(model_dir)
        if model_dir:
            model_path, mmproj_path, model_label = pick_best_model(model_dir, prefer_small=use_small, vram_gb=vram_gb)

    if not model_path:
        result["error"] = (
            "No Qwen GGUF model found and auto-download failed.\n"
            "You can manually download from: https://huggingface.co/unsloth/Qwen3.5-27B-GGUF\n"
            "  - Qwen3.5-27B-Q4_K_M.gguf\n"
            "  - mmproj-F16.gguf\n"
            "Then set: $env:QWEN_MODEL_PATH = \"your\\path\""
        )
        return result

    result["model"] = model_label

    # Check input exists
    if not os.path.exists(media_path):
        result["error"] = f"File not found: {media_path}"
        return result

    # Handle video
    frames = None
    video_info = None
    if is_video:
        frames, video_info = extract_video_frames(media_path, fps=video_fps, max_frames=video_max_frames)
        if frames is None:
            result["error"] = video_info
            return result
        result["video_info"] = video_info

    # Check dependencies — auto-install if missing
    patch_llama_cpp_winmode()  # Fix CUDA DLL loading on Windows
    try:
        from llama_cpp import Llama
    except ImportError:
        print("  [qwen] llama-cpp-python not found. Auto-installing...", file=sys.stderr)
        success = auto_install_llama_cpp()
        if not success:
            result["error"] = (
                "Auto-install of llama-cpp-python failed.\n"
                "Try manually:\n"
                "  $env:CMAKE_ARGS=\"-DGGML_CUDA=ON\"\n"
                "  pip install llama-cpp-python --force-reinstall --no-cache-dir"
            )
            return result
        patch_llama_cpp_winmode()  # Fix after fresh install too
        from llama_cpp import Llama

    try:
        t0 = time.time()
        # Determine GPU layers
        n_gpu_layers = int(os.environ.get("QWEN_GPU_LAYERS", "-1"))

        llm = Llama(
            model_path=model_path,
            mmproj=mmproj_path,
            n_ctx=int(os.environ.get("QWEN_CTX_SIZE", "8192")),
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        print(f"  [qwen] Model loaded in {time.time()-t0:.0f}s", file=sys.stderr)

        default_prompt = (
            "Please describe this image in detail. "
            "What is the content, style, composition, and any text visible? "
            "If there are people, describe their appearance and actions. "
            "If there is text, read it verbatim."
        )

        if is_video:
            # Video: process each frame
            question = prompt or (
                "Describe what is happening in this video frame. "
                "What objects, people, and actions do you see?"
            )
            frame_results = []
            for f_idx, pil_img in frames:
                tf = time.time()
                # Save frame to temp file for llama.cpp
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name
                    pil_img.save(tmp_path, format="JPEG", quality=85)

                try:
                    output = llm.create_chat_completion(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": tmp_path}},
                                    {"type": "text", "text": question},
                                ],
                            }
                        ],
                        max_tokens=512,
                        temperature=0.1,
                    )
                    frame_text = output["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    frame_text = f"[error: {e}]"
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

                frame_results.append({
                    "frame_index": f_idx,
                    "analysis": frame_text,
                    "time_s": round(time.time() - tf, 1),
                })
                print(f"  [qwen] Frame {f_idx+1}/{len(frames)} done in {frame_results[-1]['time_s']}s", file=sys.stderr)

            result["frame_analyses"] = frame_results

            # Generate summary if multiple frames
            if len(frame_results) > 1:
                summaries = [f["analysis"] for f in frame_results]
                combined = "\n---\n".join(summaries)
                result["analysis"] = (
                    f"[Video analysis: {len(frame_results)} frames from {video_info['total_frames']} total frames]\n\n"
                    f"Frame-by-frame descriptions:\n{combined}"
                )
            elif frame_results:
                result["analysis"] = frame_results[0]["analysis"]
        else:
            # Image: single analysis
            output = llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": media_path}},
                            {"type": "text", "text": prompt or default_prompt},
                        ],
                    }
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            result["analysis"] = output["choices"][0]["message"]["content"].strip()

        result["total_time_s"] = round(time.time() - t0, 1)

    except Exception as e:
        result["error"] = f"Qwen analysis failed: {e}"
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


# ---------------------------------------------------------------------------
# Auto-setup: install dependencies and download model without user intervention
# ---------------------------------------------------------------------------

def patch_llama_cpp_winmode():
    """Fix llama_cpp winmode issue on Windows.

    The llama_cpp package sets winmode=ctypes.RTLD_GLOBAL when loading llama.dll,
    which restricts DLL search paths and causes WinError 1920 on CUDA builds.
    This patches the installed package to use winmode=None instead.
    """
    try:
        import site
        site_packages = site.getsitepackages()
        if not site_packages:
            site_packages = [os.path.dirname(os.path.dirname(os.__file__))]
        candidates = [os.path.join(sp, "llama_cpp", "_ctypes_extensions.py") for sp in site_packages]
        if hasattr(site, 'getusersitepackages'):
            candidates.append(os.path.join(site.getusersitepackages(), "llama_cpp", "_ctypes_extensions.py"))

        ext_path = None
        for c in candidates:
            if os.path.exists(c):
                ext_path = c
                break

        if ext_path:
            with open(ext_path, "r") as f:
                content = f.read()
            if "cdll_args[\"winmode\"] = ctypes.RTLD_GLOBAL" in content:
                new = content.replace(
                    'cdll_args["winmode"] = ctypes.RTLD_GLOBAL',
                    'cdll_args["winmode"] = None'
                )
                with open(ext_path, "w") as f:
                    f.write(new)
                print("  [qwen] Patched llama_cpp winmode for CUDA compatibility.", file=sys.stderr)

        # Also add CUDA bin to DLL search path
        cuda_path = os.environ.get("CUDA_PATH", "")
        if cuda_path and os.path.isdir(os.path.join(cuda_path, "bin")):
            os.add_dll_directory(os.path.join(cuda_path, "bin"))
    except Exception:
        pass  # Patch is best-effort


def auto_install_llama_cpp():
    """Auto-install llama-cpp-python with CUDA or CPU backend based on hardware."""
    import subprocess, sys

    # Detect CUDA
    cuda_avail = False
    vram = 0
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        pass

    if cuda_avail and vram >= 6:
        print(f"  [qwen] Installing llama-cpp-python with CUDA (GPU: {vram}GB VRAM)...", file=sys.stderr)
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_CUDA=ON"
        # Try pre-built CUDA wheel first (fast), fall back to source build
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "llama-cpp-python",
                 "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124"],
                env=env, capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                print("  [qwen] llama-cpp-python installed with CUDA (pre-built wheel).", file=sys.stderr)
                return True
        except Exception:
            pass
        # Fallback: source build
        print("  [qwen] Pre-built wheel not found. Compiling from source (may take 10-15 min)...", file=sys.stderr)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "llama-cpp-python",
                 "--force-reinstall", "--no-cache-dir"],
                env=env, capture_output=True, text=True, timeout=900
            )
            if result.returncode == 0:
                print("  [qwen] llama-cpp-python compiled with CUDA successfully.", file=sys.stderr)
                return True
            print(f"  [qwen] Source build failed: {result.stderr[-300:]}", file=sys.stderr)
            return False
        except subprocess.TimeoutExpired:
            print("  [qwen] Source build timed out after 15 minutes.", file=sys.stderr)
            return False
    else:
        print("  [qwen] Installing llama-cpp-python (CPU version)...", file=sys.stderr)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "llama-cpp-python"],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                print("  [qwen] llama-cpp-python installed (CPU).", file=sys.stderr)
                return True
            return False
        except Exception as e:
            print(f"  [qwen] Install failed: {e}", file=sys.stderr)
            return False


def auto_download_model(model_dir=None):
    """Auto-download Qwen3.5-27B GGUF model from HuggingFace.

    Returns the model directory path if successful, None otherwise.
    """
    # Safety: refuse download if VRAM/RAM insufficient
    vram = 0
    ram = 0
    try:
        import torch
        if torch.cuda.is_available():
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        pass
    try:
        import psutil
        ram = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    if vram > 0 and vram < 6 and ram < 16:
        print(f"  [qwen] SAFETY: VRAM {vram}GB, RAM {ram}GB — too low for Qwen GGUF. "
              f"Need >=6GB VRAM or >=16GB RAM. Refusing download.", file=sys.stderr)
        return None
    if vram > 0 and vram < 12 and ram < 32:
        # Low VRAM but enough RAM — warn but allow CPU mode
        print(f"  [qwen] VRAM {vram}GB below 12GB threshold. Qwen will run on CPU (slow).", file=sys.stderr)

    # Determine target directory
    if model_dir and os.path.isdir(model_dir):
        target = model_dir
    else:
        env_dir = os.environ.get("QWEN_MODEL_PATH")
        if env_dir and os.path.isdir(env_dir):
            target = env_dir
        else:
            # Default: create alongside skill
            target = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Qwen3.5-27B")

    os.makedirs(target, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download, HfApi
    except ImportError:
        print("  [qwen] huggingface_hub not available. Installing...", file=sys.stderr)
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], capture_output=True)
        try:
            from huggingface_hub import hf_hub_download, HfApi
        except ImportError:
            return None

    repo = "unsloth/Qwen3.5-27B-GGUF"
    files_to_download = ["mmproj-F16.gguf", "Qwen3.5-27B-Q4_K_M.gguf"]

    for fname in files_to_download:
        dest = os.path.join(target, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000_000:
            print(f"  [qwen] {fname} already exists, skipping.", file=sys.stderr)
            continue

        # Retry loop for network issues
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            print(f"  [qwen] Downloading {fname} (attempt {attempt}/{max_retries})...", file=sys.stderr)
            try:
                hf_hub_download(repo, fname, local_dir=target, local_dir_use_symlinks=False)
                size_gb = os.path.getsize(dest) / (1024**3) if os.path.exists(dest) else 0
                print(f"  [qwen] {fname} downloaded ({size_gb:.1f} GB).", file=sys.stderr)
                break  # Success
            except Exception as e:
                err_str = str(e)
                if "IncompleteRead" in err_str and attempt < max_retries:
                    print(f"  [qwen] Connection dropped, retrying in {attempt * 10}s...", file=sys.stderr)
                    time.sleep(attempt * 10)
                else:
                    print(f"  [qwen] Download failed for {fname}: {e}", file=sys.stderr)
                    return None

    print(f"  [qwen] Model ready at: {target}", file=sys.stderr)
    return target


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: qwen_analyze.py <media_path> [prompt] [--model-path <dir>] [--use-small] [--video]"}))
        sys.exit(1)

    path = sys.argv[1]
    prompt = None
    model_dir = None
    use_small = False
    is_video = False

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--model-path" and i + 1 < len(sys.argv):
            model_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--use-small":
            use_small = True
            i += 1
        elif sys.argv[i] == "--video":
            is_video = True
            i += 1
        elif not sys.argv[i].startswith("--"):
            prompt = sys.argv[i]
            i += 1
        else:
            i += 1

    result = analyze_media(path, prompt, model_dir, use_small, is_video)
    print(json.dumps(result, ensure_ascii=False, indent=2))
