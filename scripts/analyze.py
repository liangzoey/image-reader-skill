#!/usr/bin/env python3
"""
All-in-one image/video analysis: auto-detects hardware, runs OCR + VLM analysis.

Usage:
    python analyze.py <image_path> [prompt]
    python analyze.py <video_path> [prompt] --mode qwen
    python analyze.py <image_path> --mode janus --use-small
    python analyze.py <path> --mode qwen --qwen-model-path <gguf_dir>
"""
import sys, os, json, time, logging, warnings
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(SKILL_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from auto_detect import detect, is_video_file

MODEL_7B = "deepseek-ai/Janus-Pro-7B"
MODEL_1B = "deepseek-ai/Janus-Pro-1B"

def check_model_cached(model_id):
    """Check if a HuggingFace model is already cached locally."""
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == model_id:
                revisions = list(repo.revisions)
                if revisions:
                    total = sum(r.size_on_disk for r in revisions)
                    return total > 1_000_000_000
    except ImportError:
        pass
    hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    model_path = os.path.join(hf_home, "hub", f"models--{model_id.replace('/', '--')}")
    if os.path.exists(model_path):
        total = 0
        for dirpath, _, filenames in os.walk(model_path):
            for f in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        return total > 500_000_000
    return False


# Minimum hardware requirements (GB)
_HW_REQUIREMENTS = {
    "janus_7b": {"vram": 14, "ram": 0},     # Janus-Pro-7B FP16
    "janus_1b": {"vram": 4, "ram": 16},     # Janus-Pro-1B
    "qwen_27b": {"vram": 12, "ram": 32},    # Qwen3.5-27B GGUF
    "qwen_14b": {"vram": 8, "ram": 20},     # Qwen2.5-14B GGUF
    "qwen_7b":  {"vram": 6, "ram": 16},     # Qwen2.5-7B GGUF
}


def _hw_safe(hw, model_key):
    """Check if hardware meets minimum requirements for a model.
    Returns True if safe, False if should refuse."""
    vram = hw.get("vram_total_gb") or 0
    ram = hw.get("ram_total_gb") or 0
    req = _HW_REQUIREMENTS.get(model_key)
    if not req:
        return False
    # GPU path: check VRAM
    if hw.get("cuda_available") and vram >= req["vram"]:
        return True
    # CPU path: check RAM
    if not hw.get("cuda_available") and ram >= req["ram"]:
        return True
    # GPU but VRAM insufficient
    if hw.get("cuda_available") and vram < req["vram"]:
        return False
    # No GPU and RAM insufficient
    return False


def analyze(media_path, prompt=None, force_mode=None, model_path=None, use_small=False,
            qwen_model_path=None, video_mode=None, video_fps=1.0, video_max_frames=30):
    t0 = time.time()
    is_vid = video_mode if video_mode is not None else is_video_file(media_path)
    hw = detect(model_path_hint=qwen_model_path)

    result = {
        "file": media_path,
        "hardware": hw,
        "mode_used": None,
        "ocr_text": None,
        "structure": None,
        "description": None,
        "video_analysis": None,
        "video_info": None,
        "error": None,
    }

    vram = hw.get("vram_total_gb") or 0
    ram = hw.get("ram_total_gb") or 0
    cuda = hw.get("cuda_available", False)

    # -- Always run OCR for images (skip for video) --
    if not is_vid:
        try:
            from ocr_image import analyze_image as ocr_analyze
            ocr_result = ocr_analyze(media_path)
            result["ocr_text"] = ocr_result.get("text_found")
            result["structure"] = ocr_result.get("structure")
        except Exception as e:
            result["warning"] = f"OCR skipped: {e}"

    # -- Determine mode priority --
    # Forced mode always wins
    if force_mode:
        selected_mode = force_mode
    elif is_vid:
        # Video: Qwen is the only option
        selected_mode = hw.get("recommended_video_mode", "ocr")
    else:
        # Image: Janus first (lightweight), Qwen as fallback
        selected_mode = hw.get("recommended_image_mode", "ocr")

    result["mode_used"] = selected_mode

    # -- Run Qwen mode (images + video) --
    if selected_mode == "qwen":
        # Safety check: refuse if hardware can't handle it
        qwen_safe = _hw_safe(hw, "qwen_27b") or _hw_safe(hw, "qwen_14b") or _hw_safe(hw, "qwen_7b")
        if not qwen_safe:
            result["error"] = (
                f"Safety: insufficient hardware for Qwen GGUF model. "
                f"VRAM: {vram}GB, RAM: {ram}GB, CUDA: {cuda}. "
                f"Need >=12GB VRAM (GPU) or >=32GB RAM (CPU). Refusing to download."
            )
            result["mode_used"] = "refused"
            result["total_time_s"] = round(time.time() - t0, 1)
            return result
        try:
            from qwen_analyze import analyze_media as qwen_analyze
            q_result = qwen_analyze(
                media_path, prompt=prompt,
                model_dir=qwen_model_path,
                use_small=use_small,
                is_video=is_vid,
                video_fps=video_fps,
                video_max_frames=video_max_frames,
            )
            result["description"] = q_result.get("analysis")
            result["video_analysis"] = q_result.get("frame_analyses")
            result["video_info"] = q_result.get("video_info")
            if q_result.get("error"):
                result["error"] = q_result["error"]
        except Exception as e:
            result["error"] = f"Qwen analysis failed: {e}"

    # -- Run Janus mode (images only, lightweight) --
    elif selected_mode == "janus" and not is_vid:
        janus_model_id = model_path or (MODEL_1B if use_small else MODEL_7B)
        # Safety check: refuse if hardware can't handle Janus
        is_7b = "7B" in janus_model_id
        janus_safe = _hw_safe(hw, "janus_7b") if is_7b else _hw_safe(hw, "janus_1b")
        if not janus_safe:
            result["error"] = (
                f"Safety: insufficient hardware for Janus-{'7B' if is_7b else '1B'}. "
                f"VRAM: {vram}GB, RAM: {ram}GB, CUDA: {cuda}. "
                f"Need {'>=14GB VRAM (GPU)' if is_7b else '>=4GB VRAM or >=16GB RAM'}. "
                f"Refusing to download."
            )
            result["mode_used"] = "refused"
            result["total_time_s"] = round(time.time() - t0, 1)
            return result
        janus_available = check_model_cached(janus_model_id) if not model_path else True

        if janus_available or (model_path and os.path.exists(model_path)):
            try:
                from janus_analyze import analyze_image as janus_analyze
                j_result = janus_analyze(media_path, prompt=prompt, model_path=model_path, use_small=use_small)
                result["description"] = j_result.get("analysis")
                if j_result.get("error"):
                    result["error"] = j_result["error"]
            except Exception as e:
                # Janus failed — try Qwen fallback (if available)
                if hw.get("qwen_available"):
                    result["warning"] = f"Janus failed ({e}), falling back to Qwen."
                    result["mode_used"] = "qwen_fallback"
                    try:
                        from qwen_analyze import analyze_media as qwen_analyze
                        q_result = qwen_analyze(media_path, prompt=prompt, model_dir=qwen_model_path,
                                                use_small=use_small, is_video=False)
                        result["description"] = q_result.get("analysis")
                        if q_result.get("error"):
                            result["error"] = q_result["error"]
                    except Exception as e2:
                        result["error"] = f"Janus failed ({e}), Qwen fallback also failed ({e2})"
                else:
                    result["error"] = f"Janus analysis failed: {e}"
        else:
            # Janus not cached — try Qwen fallback
            if hw.get("qwen_available"):
                result["warning"] = f"Janus model not cached ({janus_model_id}), falling back to Qwen."
                result["mode_used"] = "qwen_fallback"
                try:
                    from qwen_analyze import analyze_media as qwen_analyze
                    q_result = qwen_analyze(media_path, prompt=prompt, model_dir=qwen_model_path,
                                            use_small=use_small, is_video=False)
                    result["description"] = q_result.get("analysis")
                    if q_result.get("error"):
                        result["error"] = q_result["error"]
                except Exception as e:
                    result["error"] = f"Qwen fallback failed: {e}"
            else:
                result["error"] = (
                    f"Janus model not cached ({janus_model_id}). "
                    f"Run: python -c \"from huggingface_hub import snapshot_download; "
                    f"snapshot_download('{janus_model_id}')\""
                )

    result["total_time_s"] = round(time.time() - t0, 1)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: analyze.py <image_or_video_path> [prompt]"}))
        sys.exit(1)

    media_path = sys.argv[1]
    prompt = None
    force_mode = None
    model_path = None
    use_small = False
    qwen_model_path = None
    video_mode = None
    video_fps = 1.0
    video_max_frames = 30

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--mode" and i + 1 < len(sys.argv):
            force_mode = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--model-path" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--qwen-model-path" and i + 1 < len(sys.argv):
            qwen_model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--use-small":
            use_small = True
            i += 1
        elif sys.argv[i] == "--video":
            video_mode = True
            i += 1
        elif sys.argv[i] == "--video-fps" and i + 1 < len(sys.argv):
            video_fps = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--video-max-frames" and i + 1 < len(sys.argv):
            video_max_frames = int(sys.argv[i + 1])
            i += 2
        elif not sys.argv[i].startswith("--"):
            prompt = sys.argv[i]
            i += 1
        else:
            i += 1

    if not os.path.exists(media_path):
        print(json.dumps({"error": f"File not found: {media_path}"}))
        sys.exit(1)

    result = analyze(media_path, prompt, force_mode, model_path, use_small,
                     qwen_model_path, video_mode, video_fps, video_max_frames)
    print(json.dumps(result, ensure_ascii=False, indent=2))
