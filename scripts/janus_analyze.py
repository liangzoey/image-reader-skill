#!/usr/bin/env python3
"""
Analyze images using Janus-Pro multimodal model (7B or 1B).
Provides detailed image understanding beyond basic OCR.

Usage:
    python janus_analyze.py <image_path> [prompt] [--use-small] [--model-path <path>]

Flags:
    --use-small     Use Janus-Pro-1B (~2GB) instead of 7B (~15GB)
    --model-path    Specify local model directory (overrides $env:JANUS_MODEL_PATH)
"""
import sys, os, json, time, logging, warnings
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Add janus source to path
JANUS_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "janus")
sys.path.insert(0, JANUS_SRC)

try:
    import torch
    from PIL import Image
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from janus.utils.io import load_pil_images
except ImportError:
    print(json.dumps({
        "error": "Missing dependencies",
        "hint": (
            "Install the required packages:\n"
            "  pip install torch torchvision transformers timm attrdict Pillow\n\n"
            "If using ComfyUI's Python, activate its venv first."
        )
    }, ensure_ascii=False))
    sys.exit(1)

# Model source: HuggingFace model ID (auto-downloads) or local path
# Override with env var JANUS_MODEL_PATH or --model-path argument
MODEL_7B = "deepseek-ai/Janus-Pro-7B"
MODEL_1B = "deepseek-ai/Janus-Pro-1B"

def analyze_image(img_path, prompt=None, model_path=None, use_small=False):
    if model_path:
        pass  # explicit path takes priority
    elif os.environ.get("JANUS_MODEL_PATH"):
        model_path = os.environ["JANUS_MODEL_PATH"]
    elif use_small:
        model_path = MODEL_1B
    else:
        model_path = MODEL_7B
    result = {
        "file": img_path,
        "model": "Janus-Pro-7B" if "7B" in model_path else "Janus-Pro-1B",
        "error": None,
        "analysis": None,
    }

    if not os.path.exists(img_path):
        result["error"] = f"Image not found: {img_path}"
        return result

    try:
        # Load processor and model
        t0 = time.time()
        vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer
        print(f"  [janus] Processor loaded in {time.time()-t0:.0f}s", file=sys.stderr)

        t1 = time.time()
        vl_gpt = MultiModalityCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        vl_gpt = vl_gpt.cuda().eval()
        print(f"  [janus] Model loaded in {time.time()-t1:.0f}s", file=sys.stderr)

        # Default prompt for general analysis
        default_prompt = (
            "Please describe this image in detail. "
            "What is the content, style, composition, and any text visible in the image? "
            "If there are people, describe their appearance, pose, and expression. "
            "If there is text, read it verbatim."
        )

        question = prompt or default_prompt

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [img_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)

        # Run image encoder
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # Generate response
        t2 = time.time()
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
        )
        print(f"  [janus] Generation took {time.time()-t2:.1f}s", file=sys.stderr)

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        result["analysis"] = answer.strip()
        result["total_time_s"] = round(time.time() - t0, 1)

    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: janus_analyze.py <image_path> [prompt]"}))
        sys.exit(1)

    path = sys.argv[1]
    prompt = None
    model_path = None
    use_small = False
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--model-path" and i + 1 < len(sys.argv):
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

    result = analyze_image(path, prompt, model_path, use_small)
    print(json.dumps(result, ensure_ascii=False, indent=2))
