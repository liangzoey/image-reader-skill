#!/usr/bin/env python3
"""
All-in-one image analysis: auto-detects hardware, checks model cache, runs best mode.

Usage:
    python analyze.py <image_path> [prompt]
    python analyze.py <image_path> --mode janus --use-small
"""
import sys, os, json, time, logging, warnings
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(SKILL_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from auto_detect import detect

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
                    return total > 1_000_000_000  # at least 1GB cached
    except ImportError:
        pass
    # Fallback: check standard HF cache path
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


def analyze(img_path, prompt=None, force_mode=None, model_path=None, use_small=False):
    t0 = time.time()
    hw = detect()
    mode = force_mode or hw["recommended_mode"]

    result = {
        "file": img_path,
        "hardware": hw,
        "mode_used": mode,
        "ocr_text": None,
        "structure": None,
        "description": None,
        "error": None,
    }

    # Always run OCR (fast, structural info)
    try:
        from ocr_image import analyze_image as ocr_analyze
        ocr_result = ocr_analyze(img_path)
        result["ocr_text"] = ocr_result.get("text_found")
        result["structure"] = ocr_result.get("structure")
    except Exception as e:
        result["warning"] = f"OCR skipped: {e}"

    # Run Janus if hardware + model availability allows
    if mode == "janus" or force_mode == "janus":
        janus_model_id = model_path or (MODEL_1B if use_small else MODEL_7B)

        if model_path and os.path.exists(model_path):
            pass  # explicit local path
        elif not check_model_cached(janus_model_id):
            result["error"] = (
                f"Model not cached ({janus_model_id}). "
                f"Run this to download it first:\n"
                f"  python -c \"from huggingface_hub import snapshot_download; snapshot_download('{janus_model_id}')\""
            )
            result["total_time_s"] = round(time.time() - t0, 1)
            return result

        try:
            from janus_analyze import analyze_image as janus_analyze
            j_result = janus_analyze(img_path, prompt=prompt, model_path=model_path, use_small=use_small)
            result["description"] = j_result.get("analysis")
            if j_result.get("error"):
                result["error"] = j_result["error"]
        except Exception as e:
            result["error"] = f"Janus analysis failed: {e}"

    result["total_time_s"] = round(time.time() - t0, 1)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: analyze.py <image_path> [prompt]"}))
        sys.exit(1)

    img_path = sys.argv[1]
    prompt = None
    force_mode = None
    model_path = None
    use_small = False

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--mode" and i + 1 < len(sys.argv):
            force_mode = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--model-path" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--use-small":
            use_small = True
            i += 1
        elif not sys.argv[i].startswith("--"):
            prompt = sys.argv[i]
            i += 1
        else:
            i += 1

    if not os.path.exists(img_path):
        print(json.dumps({"error": f"Image not found: {img_path}"}))
        sys.exit(1)

    result = analyze(img_path, prompt, force_mode, model_path, use_small)
    print(json.dumps(result, ensure_ascii=False, indent=2))
