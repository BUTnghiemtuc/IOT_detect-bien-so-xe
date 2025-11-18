# ocr_bridge.py (improved: dùng detect+OCR từ detect_plate.py)

import struct, time, re
import requests
import serial
import cv2
import numpy as np
from typing import List, Tuple, Optional

# ====== Cấu hình ======
SERIAL_PORT = "COM3"                 # đổi đúng cổng của bạn
BAUD = 115200
API_BASE = "http://127.0.0.1:8000"   # đổi sang IP server FastAPI nếu chạy máy khác
POST_URL = f"{API_BASE}/events"

MAGIC = 0x50494354  # 'PICT'
MAX_FRAME = 3 * 1024 * 1024  # 3MB an toàn cho JPEG

# ====== Model & OCR config (theo detect_plate.py) ======
MODEL_PATH = "license_plate_detector.pt"      # hoặc weights fine-tuned biển số
CONF_THR = 0.25
IOU_THR  = 0.45
OCR_LANG = "en"                      # "vi" nếu bạn muốn, khi model Paddle hỗ trợ
USE_PADDLE = True

# ====== Import lại các helpers từ detect_plate.py ======
# (Các hàm này được định nghĩa trong detect_plate.py của bạn)
from ultralytics import YOLO

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

def get_paddle_ocr(lang: str = "en"):
    if not PADDLE_AVAILABLE:
        raise RuntimeError("PaddleOCR chưa cài. pip install paddleocr")
    return PaddleOCR(lang=lang, use_angle_cls=True)

def ocr_with_paddle(ocr, image: np.ndarray) -> str:
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = ocr.ocr(img_rgb, cls=True)
    texts = []
    for line in result:
        for rec in line:
            if len(rec) >= 2:
                txt = rec[1][0]
                texts.append(txt.strip())
    return " ".join(texts).strip()

def ocr_with_tesseract(image: np.ndarray) -> str:
    try:
        import pytesseract
    except Exception:
        raise RuntimeError("pytesseract chưa cài. pip install pytesseract (và cài tesseract-ocr hệ thống)")
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(th, config="--psm 7")
    return text.strip()

# ====== Khởi tạo model/OCR một lần ======
_yolo_model: Optional[YOLO] = None
_paddle = None

def init_models():
    global _yolo_model, _paddle
    if _yolo_model is None:
        _yolo_model = YOLO(MODEL_PATH)  # ultralytics hỗ trợ predict trên numpy BGR
    if USE_PADDLE and PADDLE_AVAILABLE and _paddle is None:
        try:
            _paddle = get_paddle_ocr(OCR_LANG)
        except Exception as e:
            print("[OCR] Không khởi tạo được PaddleOCR:", e)
            _paddle = None

# ====== Tiện ích ======
PLATE_KEEP = re.compile(r"[A-Z0-9]")
def normalize_plate(s: str) -> str:
    s_up = s.upper()
    kept = "".join(ch for ch in s_up if PLATE_KEEP.match(ch))
    return kept[:12] if kept else ""

def enhance_crop(crop: np.ndarray) -> np.ndarray:
    # Resize nếu quá nhỏ
    Hc, Wc = crop.shape[:2]
    if min(Hc, Wc) < 40:
        scale = max(1, int(80 / max(Hc, Wc)))
        crop = cv2.resize(crop, (Wc * scale, Hc * scale), interpolation=cv2.INTER_CUBIC)

    # CLAHE trên kênh L (LAB)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    crop_enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return crop_enh

def best_plate_from_detections(texts: List[str]) -> str:
    # Ưu tiên chuỗi hợp lệ dài hơn (tối đa 12), fallback UNKNOWN
    candidates = [normalize_plate(t) for t in texts if t]
    if not candidates:
        return "UNKNOWN"
    # chọn theo độ dài; nếu bằng nhau thì chọn cái đầu
    candidates.sort(key=lambda x: len(x), reverse=True)
    return candidates[0] if len(candidates[0]) >= 4 else "UNKNOWN"

# ====== Chạy detect + OCR trên frame numpy ======
def detect_and_ocr_frame(img_bgr: np.ndarray) -> Tuple[str, list]:
    """
    Trả về: (plate_text_best, list_detections)
    list_detections: [{box:[x1,y1,x2,y2], text:str, conf:float}, ...]
    """
    init_models()
    h, w = img_bgr.shape[:2]
    results = _yolo_model.predict(source=img_bgr, conf=CONF_THR, iou=IOU_THR, verbose=False)
    r = results[0]

    boxes = []
    if hasattr(r, "boxes") and len(r.boxes) > 0:
        for box in r.boxes:
            xy = box.xyxy[0].cpu().numpy().astype(int)
            conf_val = float(box.conf[0].cpu().numpy()) if hasattr(box, "conf") else None
            x1, y1, x2, y2 = xy.tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            boxes.append((x1, y1, x2, y2, conf_val))

    detections = []
    texts_collected = []

    # Nếu không có box, có thể thêm heuristic như detect_plate.py — bỏ qua để gọn
    for (x1, y1, x2, y2, conf_val) in boxes:
        crop = img_bgr[y1:y2, x1:x2].copy()
        crop_enh = enhance_crop(crop)

        text = ""
        # Thử Paddle trước
        if _paddle is not None:
            try:
                text = ocr_with_paddle(_paddle, crop_enh)
            except Exception as ex:
                print("[OCR] Paddle lỗi:", ex)
                text = ""
        # Fallback tesseract
        if not text:
            try:
                text = ocr_with_tesseract(crop_enh)
            except Exception as ex:
                print("[OCR] Tesseract lỗi:", ex)
                text = ""

        detections.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "conf": conf_val,
            "text": text
        })
        texts_collected.append(text)

    best_text = best_plate_from_detections(texts_collected)
    return best_text, detections

# ====== Serial helpers ======
def read_exact(ser: serial.Serial, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)

def run():
    print(f"[SER] open {SERIAL_PORT}@{BAUD}")
    ser = serial.Serial(SERIAL_PORT, baudrate=BAUD, timeout=3)

    while True:
        try:
            hdr = read_exact(ser, 4)
            if len(hdr) < 4:
                continue
            magic = struct.unpack("<I", hdr)[0]
            if magic != MAGIC:
                # trôi khung -> bỏ 1 byte & tiếp
                ser.read(1)
                continue

            le = read_exact(ser, 4)
            if len(le) < 4:
                continue
            L = struct.unpack("<I", le)[0]
            if L <= 0 or L > MAX_FRAME:
                print("[FRAME] bad length:", L)
                continue

            jpg = read_exact(ser, L)
            if len(jpg) < L:
                print("[FRAME] truncated")
                continue

            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print("[OCR] decode fail")
                continue

            # === Detect + OCR theo detect_plate.py ===
            plate, dets = detect_and_ocr_frame(img)
            print(f"[OCR] plate = {plate} | dets={len(dets)}")

            payload = {"plate": plate, "detections": dets}
            r = requests.post(POST_URL, json=payload, timeout=5)
            print("[API]", r.status_code, r.text[:200])

        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print("[ERR]", e)
            time.sleep(0.5)

# if __name__ == "__main__":
#     run()


if __name__ == "__main__":
    # Test OCR offline (không cần ESP32)
    import glob, cv2

    print("=== TEST OCR LOCAL ===")
    image_files = sorted(glob.glob("images/*.jpg"))
    for img_path in image_files:
        print(f"\n🔍 Ảnh: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print("Không đọc được ảnh!")
            continue

        plate_text, detections = detect_and_ocr_frame(img)
        print("Kết quả OCR:", plate_text)
        for det in detections:
            print("  Box:", det["box"], "| Text:", det["text"])

    print("\n=== HOÀN TẤT TEST OCR ===")
