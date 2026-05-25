#!/usr/bin/env python3
"""
Analyze an image using OpenCV + EasyOCR.
Outputs structured information about image content, text, and layout.
"""
import sys, json, os, logging
logging.getLogger().setLevel(logging.ERROR)
# Don't set TORCH_LOGS to empty string - it breaks torch._logging

import cv2
import numpy as np
from pathlib import Path

def describe_color(bgr):
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    if r > 200 and g > 200 and b > 200:
        return "white"
    if r < 30 and g < 30 and b < 30:
        return "black/near-black"
    if r > 200 and g < 100 and b < 100:
        return "red"
    if r > 200 and g > 150 and b < 100:
        return "orange/yellow"
    if r < 100 and g > 180 and b < 100:
        return "green"
    if r < 100 and g < 100 and b > 180:
        return "blue"
    if r > 150 and g > 150 and b < 150:
        return "yellow/gold"
    if r > 100 and g < 80 and b > 100:
        return "purple/magenta"
    if abs(r-g) < 20 and abs(g-b) < 20:
        return f"gray({r})"
    return f"rgb({r},{g},{b})"

def analyze_image(img_path):
    result = {"file": img_path, "error": None, "text_found": [], "structure": {}}

    img = cv2.imread(str(img_path))
    if img is None:
        result["error"] = "Cannot read image file"
        return result

    h, w, c = img.shape
    result["structure"]["dimensions"] = f"{w}x{h}"
    result["structure"]["aspect_ratio"] = round(w / h, 3)
    result["structure"]["channels"] = c
    result["structure"]["file_size_bytes"] = os.path.getsize(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Overall stats
    result["structure"]["brightness"] = round(float(gray.mean()), 1)
    result["structure"]["contrast"] = round(float(gray.std()), 1)

    # Edge density (indicates text/detail)
    edges = cv2.Canny(gray, 50, 150)
    edge_pct = round(float((edges > 0).sum() / edges.size * 100), 1)
    result["structure"]["edge_density_percent"] = edge_pct

    # Dominant colors (simplified: sample grid)
    sampling = img[::max(1, h//20), ::max(1, w//20)]
    flat = sampling.reshape(-1, 3)
    unique_colors = min(10000, len(np.unique(flat, axis=0)))
    result["structure"]["estimated_unique_colors"] = unique_colors

    # Region grid analysis (4x4 grid)
    regions = []
    for row in range(4):
        for col in range(4):
            x1, x2 = col * w // 4, (col + 1) * w // 4
            y1, y2 = row * h // 4, (row + 1) * h // 4
            roi = gray[y1:y2, x1:x2]
            avg_bgr = img[y1 + (y2-y1)//2, x1 + (x2-x1)//2]
            region_edges = cv2.Canny(roi, 50, 150).mean()
            regions.append({
                "row": row, "col": col,
                "brightness": round(float(roi.mean()), 1),
                "contrast": round(float(roi.std()), 1),
                "edge_density": round(float(region_edges), 1),
                "dominant_color": describe_color(avg_bgr)
            })
    result["structure"]["grid_4x4"] = regions

    # Determine image type heuristics
    bright_pixels = (gray > 200).sum() / gray.size * 100
    dark_pixels = (gray < 30).sum() / gray.size * 100
    colorful = unique_colors > 5000

    if edge_pct < 2 and bright_pixels > 60:
        result["structure"]["likely_type"] = "screenshot/UI with mostly white background"
    elif edge_pct < 3 and dark_pixels > 50:
        result["structure"]["likely_type"] = "dark-mode UI / screenshot"
    elif edge_pct > 8 and colorful:
        result["structure"]["likely_type"] = "photograph / complex image"
    elif edge_pct > 5:
        result["structure"]["likely_type"] = "document / text-heavy image"
    elif bright_pixels > 30:
        result["structure"]["likely_type"] = "simple graphic / illustration"
    else:
        result["structure"]["likely_type"] = "mixed content / UI"

    # OCR with EasyOCR
    try:
        import easyocr
        # Try with multiple preprocessing
        img_big = cv2.resize(img, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
        gray_big = cv2.cvtColor(img_big, cv2.COLOR_BGR2GRAY)

        # Preprocess variations
        preprocessed = {
            "original": img_big,
            "grayscale": gray_big,
            "otsu": cv2.threshold(gray_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            "otsu_inv": cv2.threshold(cv2.bitwise_not(gray_big), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        }

        reader = easyocr.Reader(["ch_sim", "en"], gpu=True, verbose=False)

        all_texts = {}
        for method_name, processed_img in preprocessed.items():
            temp_path = None
            try:
                temp_path = f"_ocr_temp_{method_name}.png"
                cv2.imwrite(temp_path, processed_img)
                ocr_result = reader.readtext(temp_path, detail=1, paragraph=False, width_ths=0.7, low_text=0.2)
                texts = [(text, round(conf, 2)) for _, text, conf in ocr_result if conf > 0.15]
                if texts:
                    all_texts[method_name] = texts
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        if all_texts:
            # Deduplicate texts across methods
            seen = set()
            for method, texts in all_texts.items():
                for text, conf in texts:
                    if text not in seen:
                        seen.add(text)
                        result["text_found"].append({"text": text, "confidence": conf, "method": method})
        else:
            result["text_found"] = None

    except ImportError:
        result["text_error"] = "easyocr not installed"
    except Exception as e:
        result["text_error"] = str(e)

    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: ocr_image.py <image_path>"}))
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(json.dumps({"error": f"File not found: {path}"}))
        sys.exit(1)

    result = analyze_image(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
