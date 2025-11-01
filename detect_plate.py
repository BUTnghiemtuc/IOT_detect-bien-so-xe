"""
detect_plate.py
- Dùng YOLOv8 để detect biển số
- Dùng PaddleOCR để đọc chữ trên biển
- Xử lý 1 file ảnh hoặc cả folder
- Lưu crop vào ./outputs/crops/ và kết quả JSON ./outputs/results.json

Cách dùng:
  python detect_plate.py --model yolov8n_plate.pt --source ./images --out outputs
Nếu bạn chưa có weights fine-tuned cho biển số, đặt --model yolov8n.pt (pretrained base)
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# PaddleOCR (fast, nhiều ngôn ngữ). Nếu không muốn cài PaddleOCR, có thể chuyển sang pytesseract.
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

# ---- Model cache helper ----
_MODEL_CACHE = {}

def get_yolo_model(model_path: str):
    """
    Load YOLO model once and cache it.
    model_path can be: 'yolov8n.pt' or path to your custom weights
    """
    global _MODEL_CACHE
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    model = YOLO(model_path)
    _MODEL_CACHE[model_path] = model
    return model

# ---- OCR helper ----
def get_paddle_ocr(lang: str = "en"):
    """
    Return PaddleOCR instance. lang example: "en", "vi" (if model available).
    Note: first time init may download model files.
    """
    if not PADDLE_AVAILABLE:
        raise RuntimeError("PaddleOCR không được cài. Chạy: pip install paddleocr")
    # use_angle_cls=False (faster) ; use_gpu=False by default
    return PaddleOCR(lang=lang, use_angle_cls=True)

def ocr_with_paddle(ocr, image: np.ndarray) -> str:
    """
    image: BGR numpy array (OpenCV). Returns concatenated text.
    """
    # PaddleOCR expects RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = ocr.ocr(img_rgb, cls=True)
    texts = []
    for line in result:
        # result format: [ [ [box], (text, confidence) ], ... ]
        for rec in line:
            if len(rec) >= 2:
                txt, conf = rec[1][0], rec[1][1]
                texts.append(txt.strip())
    return " ".join(texts).strip()

# Fallback OCR using pytesseract (nếu người dùng muốnn)
def ocr_with_tesseract(image: np.ndarray) -> str:
    try:
        import pytesseract
    except Exception:
        raise RuntimeError("pytesseract chưa cài. Chạy: pip install pytesseract và cấu hình tesseract-ocr trên hệ thống")
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # threshold to improve legibility
    _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(th, config='--psm 7')  # PSM 7: treat as a single text line
    return text.strip()

# ---- Detection + OCR pipeline ----
def detect_and_ocr(model_path: str, image_path: str, conf: float = 0.25, iou: float = 0.45,
                   ocr_lang: str = "en", use_paddle: bool = True) -> List[Dict]:
    """
    Returns a list of detections: [{ 'box': [x1,y1,x2,y2], 'conf': f, 'crop_path': str, 'text': str }, ...]
    """
    model = get_yolo_model(model_path)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {image_path}")
    h, w = img_bgr.shape[:2]

    # Predict with ultralytics YOLO
    # Nếu model của bạn train 1 class 'plate' thì cls id mapping sẽ là r.names
    results = model.predict(source=image_path, conf=conf, iou=iou, verbose=False, save=False)
    r = results[0]

    # Prepare OCR
    ocr = None
    if use_paddle:
        if PADDLE_AVAILABLE:
            ocr = get_paddle_ocr(lang=ocr_lang)
        else:
            print("⚠️ PaddleOCR không cài — fallback sẽ dùng pytesseract nếu có.")
            ocr = None

    detections = []
    out_dir = Path("outputs") / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    # r.boxes may be empty; iterate
    boxes = []
    if hasattr(r, "boxes") and len(r.boxes) > 0:
        for i, box in enumerate(r.boxes):
            # box.xyxy is tensor [[x1,y1,x2,y2]]
            xy = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].cpu().numpy()) if hasattr(box, "cls") else None
            conf_val = float(box.conf[0].cpu().numpy()) if hasattr(box, "conf") else None
            x1, y1, x2, y2 = xy.tolist()
            # clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            boxes.append((x1, y1, x2, y2, conf_val, cls_id))

    # If model doesn't output any boxes, try simple heuristic (optional)
    if len(boxes) == 0:
        # heuristic: try MSER/edge detection to find rectangular candidate plates (very rough)
        # (This is just a fallback, not guaranteed)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            ar = ww / (hh + 1e-6)
            area = ww * hh
            if 2000 < area < w * h * 0.5 and 2.0 < ar < 6.5:  # plate-like ratio & area
                candidates.append((x, y, x + ww, y + hh, 0.5, None))
        boxes = candidates

    # Crop each box and OCR
    for idx, (x1, y1, x2, y2, conf_val, cls_id) in enumerate(boxes):
        crop = img_bgr[y1:y2, x1:x2].copy()
        # optional: resize small crops to improve OCR
        Hc, Wc = crop.shape[:2]
        if min(Hc, Wc) < 40:
            scale = max(1, int(80 / max(Hc, Wc)))
            crop = cv2.resize(crop, (Wc * scale, Hc * scale), interpolation=cv2.INTER_CUBIC)

        # enhance contrast a little
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        crop_enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # OCR
        text = ""
        if ocr is not None:
            try:
                text = ocr_with_paddle(ocr, crop_enh)
            except Exception as ex:
                print("OCR Paddle lỗi:", ex)
                text = ""
        if (not text) or (not use_paddle):
            try:
                text = ocr_with_tesseract(crop_enh)
            except Exception as ex:
                # can't OCR
                text = ""

        # save crop
        crop_name = f"{Path(image_path).stem}_crop_{idx}.jpg"
        crop_path = out_dir / crop_name
        cv2.imwrite(str(crop_path), crop_enh)

        detections.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "conf": float(conf_val) if conf_val is not None else None,
            "class_id": int(cls_id) if cls_id is not None else None,
            "crop_path": str(crop_path),
            "text": text
        })

    return detections

# ---- Utility: process folder ----
def process_source(model_path: str, source: str, out_dir: str = "outputs", conf: float = 0.25, ocr_lang: str = "en", use_paddle: bool = True):
    """
    source: single image path or folder
    """
    p = Path(source)
    images = []
    if p.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            images.extend(sorted(p.glob(ext)))
    else:
        images = [p]

    results_all = {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(images, desc="Processing images"):
        try:
            detections = detect_and_ocr(model_path=str(model_path), image_path=str(img_path), conf=conf, ocr_lang=ocr_lang, use_paddle=use_paddle)
            results_all[str(img_path)] = detections
        except Exception as ex:
            print(f"❗ Lỗi xử lý {img_path}: {ex}")
            results_all[str(img_path)] = {"error": str(ex)}

    # save json results
    json_path = Path(out_dir) / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)

    print(f"Hoàn tất. Kết quả lưu ở: {json_path}")
    return results_all

# ---- CLI ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model (yolov8n.pt or custom weights)")
    parser.add_argument("--source", type=str, required=True, help="Image file or folder")
    parser.add_argument("--out", type=str, default="outputs", help="Output folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--ocr_lang", type=str, default="en", help="OCR language (paddleocr lang code)")
    parser.add_argument("--no-paddle", action="store_true", help="Disable PaddleOCR (use fallback)")
    args = parser.parse_args()

    use_paddle = not args.no_paddle
    process_source(model_path=args.model, source=args.source, out_dir=args.out, conf=args.conf, ocr_lang=args.ocr_lang, use_paddle=use_paddle)

if __name__ == "__main__":
    main()
