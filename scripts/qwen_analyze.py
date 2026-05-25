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


MODEL_SIZES = {
    "7b": {
        "gguf_pattern": "*7B*gguf",
        "mmproj_pattern": "*7B*mmproj*gguf",
        "min_vram": 6,
    },
    "72b": {
        "gguf_pattern": "*72B*gguf",
        "mmproj_pattern": "*72B*mmproj*gguf",
        "min_vram": 20,
    },
}


def find_gguf_files(model_dir, pattern):
    """Find GGUF files in a directory matching a pattern."""
    import glob
    files = glob.glob(os.path.join(model_dir, pattern))
    files.extend(glob.glob(os.path.join(model_dir, "**", pattern), recursive=True))
    return sorted(files)


def find_qwen_model(model_dir=None, prefer_small=False):
    """Find Qwen GGUF model files in the given directory.

    Returns (model_path, mmproj_path, model_label) or (None, None, None).
    """
    search_dirs = []
    if model_dir and os.path.isdir(model_dir):
        search_dirs.append(model_dir)
    env_dir = os.environ.get("QWEN_MODEL_PATH")
    if env_dir and os.path.isdir(env_dir):
        search_dirs.append(env_dir)

    # Try model sizes: prefer 7b for small/limited VRAM, 72b otherwise
    sizes = ["7b", "72b"] if prefer_small else ["72b", "7b"]

    for d in search_dirs:
        for size_key in sizes:
            cfg = MODEL_SIZES[size_key]
            models = find_gguf_files(d, cfg["gguf_pattern"])
            mmprojs = find_gguf_files(d, cfg["mmproj_pattern"])
            # Filter out mmproj from models list
            models = [m for m in models if "mmproj" not in os.path.basename(m).lower()]
            if models and mmprojs:
                return models[0], mmprojs[0], f"Qwen2.5-VL-{size_key.upper()}"
    return None, None, None


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
        use_small: Prefer 7B model over 72B
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

    # Find model
    model_path, mmproj_path, model_label = find_qwen_model(model_dir, prefer_small=use_small)
    if not model_path:
        result["error"] = (
            "No Qwen2.5-VL GGUF model found.\n"
            "Set QWEN_MODEL_PATH or use --model-path with a directory containing:\n"
            "  - Qwen2.5-VL-7B GGUF files (e.g., *qwen2.5-vl-7b*q4_k_m.gguf)\n"
            "  - Qwen2.5-VL-7B mmproj file (e.g., *qwen2.5-vl-7b*mmproj*.gguf)\n\n"
            "Download from: https://huggingface.co/bartowski\n\n"
            "Or for the 72B model, use --model-path with the 72B GGUF files."
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

    # Load model
    try:
        from llama_cpp import Llama
    except ImportError:
        result["error"] = (
            "llama-cpp-python not installed.\n"
            "Install with:\n"
            "  pip install llama-cpp-python\n\n"
            "For CUDA support:\n"
            "  set CMAKE_ARGS=-DGGML_CUDA=ON\n"
            "  pip install llama-cpp-python --force-reinstall --no-cache-dir"
        )
        return result

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
