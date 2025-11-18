"""
detect_plate.py
- Dùng YOLOv8 để detect biển số
- Dùng PaddleOCR để đọc chữ trên biển
- Xử lý 1 file ảnh hoặc cả folder
- Lưu crop vào ./outputs/crops/ và kết quả JSON ./outputs/results.json
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

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

# ---- Default cấu hình cho bạn ----
MODEL_DEFAULT = r"D:\IOT\license_plate_detector.pt"   # model biển số của bạn
SOURCE_DEFAULT = r"D:\IOT\captures_serial"            # thư mục ảnh (ảnh từ ESP32-CAM)
OUT_DIR_DEFAULT = "outputs"
CONF_DEFAULT = 0.25
OCR_LANG_DEFAULT = "en"   # biển số chủ yếu là chữ số/latin, en là ổn
USE_PADDLE_DEFAULT = True

# ---- Model cache helper ----
_MODEL_CACHE: Dict[str, YOLO] = {}


def get_yolo_model(model_path: str):
    """
    Load YOLO model once and cache it.
    model_path can be: 'yolov8n.pt' hoặc path tới weights custom.
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
    Return PaddleOCR instance. lang ví dụ: "en", "vi" (nếu có model).
    """
    if not PADDLE_AVAILABLE:
        raise RuntimeError("PaddleOCR không được cài. Chạy: pip install paddleocr")
    # use_angle_cls=False vì ta tự xử lý xoay 0° / 180°
    return PaddleOCR(lang=lang, use_angle_cls=False)


def paddle_ocr_text_conf(ocr, image: np.ndarray):
    """
    OCR bằng Paddle, trả về (text, avg_conf).
    image: BGR (OpenCV)
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = ocr.ocr(img_rgb, cls=False)
    texts = []
    confs = []
    for line in result:
        for rec in line:
            if len(rec) >= 2:
                txt, conf = rec[1][0], rec[1][1]
                texts.append(txt.strip())
                confs.append(float(conf))
    text = " ".join(texts).strip()
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return text, avg_conf


# Fallback OCR using pytesseract (nếu cần)
def ocr_with_tesseract(image: np.ndarray) -> str:
    try:
        import pytesseract
    except Exception:
        raise RuntimeError(
            "pytesseract chưa cài. Chạy: pip install pytesseract "
            "và cài tesseract-ocr trên hệ thống"
        )
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    text = pytesseract.image_to_string(th, config="--psm 7")
    return text.strip()


def rotate_keep_size(img: np.ndarray, angle: float) -> np.ndarray:
    """Xoay ảnh quanh tâm, giữ nguyên kích thước."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def ocr_plate_multi_orient(ocr, image_bgr: np.ndarray, use_paddle: bool = True):
    """
    Thử OCR ở 2 góc (0°, 180°), chọn text có score tốt nhất.
    Trả về (best_text, best_angle_deg).
    """
    candidates = []

    for angle in (0, 180):
        img_rot = image_bgr if angle == 0 else rotate_keep_size(image_bgr, angle)

        text = ""
        score = 0.0

        # Ưu tiên PaddleOCR nếu có
        if use_paddle and ocr is not None:
            try:
                text, conf = paddle_ocr_text_conf(ocr, img_rot)
                score = conf
            except Exception:
                text, score = "", 0.0

        # Nếu chưa có text → thử Tesseract
        if not text:
            try:
                t_text = ocr_with_tesseract(img_rot)
                t_score = len(t_text) / 10.0  # score tạm theo độ dài
                if t_score > score:
                    text, score = t_text, t_score
            except Exception:
                pass

        # cộng thêm điểm dựa trên số ký tự chữ/số (giống biển số)
        plate_chars = sum(c.isalnum() for c in text)
        score += 0.05 * plate_chars

        candidates.append((score, angle, text))

    best_score, best_angle, best_text = max(candidates, key=lambda x: x[0])
    return best_text, best_angle


# ---- Chuẩn hoá & tách nhiều biển trong 1 chuỗi OCR ----
VN_PLATE_RE = re.compile(r'^[0-9]{2}[A-Z]-[0-9]{3}\.[0-9]{2}$')


def normalize_plate_token(token: str) -> str | None:
    """
    Nhận 1 token OCR (vd: 'SOA-696.96') ->
    cố gắng sửa thành biển hợp lệ (vd: '60A-696.96') hoặc trả về None.
    """
    token = token.strip().upper()
    # giữ lại chỉ A-Z, 0-9, '-', '.'
    token = re.sub(r'[^A-Z0-9\-.]', '', token)
    if not token:
        return None

    chars = list(token)

    # Vị trí phải là số trong dạng NN L - NNN . NN
    digit_positions = [0, 1, 4, 5, 6, 8, 9]

    # Các ký tự hay bị OCR nhầm
    char_digit_map = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "6",
        "B": "8",
        "G": "6",
    }

    # Sửa các vị trí đáng ra phải là số
    for i in digit_positions:
        if i < len(chars):
            c = chars[i]
            if (not c.isdigit()) and c in char_digit_map:
                chars[i] = char_digit_map[c]

    cand = "".join(chars)

    # Kiểm tra có đúng format biển VN không
    if VN_PLATE_RE.match(cand):
        return cand

    return None


def extract_plate_strings(text: str):
    """
    Nhận 1 chuỗi OCR (vd: 'SOA-696.96 36A-490.53') ->
    trả về list các biển đã chuẩn hoá: ['60A-696.96', '36A-490.53']
    """
    if not text:
        return []

    parts = re.split(r"\s+", text.upper())
    plates = []
    for part in parts:
        norm = normalize_plate_token(part)
        if norm and norm not in plates:
            plates.append(norm)
    return plates


# ---- PREPROCESS: CẮT 2 NỬA + XOAY TRÁI/PHẢI ----
def split_and_rotate_two_plates(image_bgr: np.ndarray, base_stem: str):
    """
    Ảnh đầu vào chứa 2 biển số (như ví dụ bạn gửi):
    - Cắt đôi theo chiều dọc.
    - Nửa trái xoay 90° theo chiều kim đồng hồ (để text nằm ngang).
    - Nửa phải xoay 90° ngược chiều kim đồng hồ.

    Trả về list: [("left", img_left_rotated), ("right", img_right_rotated)]
    Đồng thời lưu debug vào outputs/preprocessed/.
    """
    h, w = image_bgr.shape[:2]
    mid = w // 2

    left = image_bgr[:, :mid].copy()
    right = image_bgr[:, mid:].copy()

    # Với ảnh bạn gửi, CW cho bên trái, CCW cho bên phải là đọc đẹp nhất
    left_rot = cv2.rotate(left, cv2.ROTATE_90_CLOCKWISE)
    right_rot = cv2.rotate(right, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Lưu debug
    pre_dir = Path(OUT_DIR_DEFAULT) / "preprocessed"
    pre_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pre_dir / f"{base_stem}_left_rot.jpg"), left_rot)
    cv2.imwrite(str(pre_dir / f"{base_stem}_right_rot.jpg"), right_rot)

    return [("left", left_rot), ("right", right_rot)]


# ---- Detection + OCR pipeline ----
def detect_and_ocr(
    model_path: str,
    image_path: str | None = None,
    image_bgr: np.ndarray | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    ocr_lang: str = "en",
    use_paddle: bool = True,
    name_suffix: str = "",
) -> List[Dict]:
    """
    Chạy YOLOv8 + OCR trên 1 ảnh (đã load sẵn hoặc từ path).

    Trả về list detections:
      [
        {
          'box': [x1,y1,x2,y2],
          'conf': float | None,
          'class_id': int | None,
          'crop_path': str,
          'raw_text': str,   # full OCR chuỗi
          'text': str,       # 1 biển đã chuẩn hoá (vd: '60A-696.96')
          'angle_used': 0 hoặc 180
        },
        ...
      ]
    """
    model = get_yolo_model(model_path)

    # Load ảnh nếu chưa có
    if image_bgr is None:
        if image_path is None:
            raise ValueError("Cần image_path hoặc image_bgr")
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Không thể đọc ảnh: {image_path}")

    img_bgr = image_bgr
    h, w = img_bgr.shape[:2]

    # Predict với ultralytics YOLO (dùng ndarray, không dùng path nữa)
    results = model.predict(
        source=img_bgr, conf=conf, iou=iou, verbose=False, save=False
    )
    r = results[0]

    # Chuẩn bị OCR
    ocr = None
    if use_paddle:
        if PADDLE_AVAILABLE:
            ocr = get_paddle_ocr(lang=ocr_lang)
        else:
            print("⚠️ PaddleOCR không cài — sẽ fallback sang pytesseract nếu có.")
            ocr = None

    detections: List[Dict] = []
    out_dir = Path(OUT_DIR_DEFAULT) / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lấy boxes từ YOLO
    boxes = []
    if hasattr(r, "boxes") and len(r.boxes) > 0:
        for box in r.boxes:
            xy = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].cpu().numpy()) if hasattr(box, "cls") else None
            conf_val = (
                float(box.conf[0].cpu().numpy()) if hasattr(box, "conf") else None
            )
            x1, y1, x2, y2 = xy.tolist()
            # clamp vào trong ảnh
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            boxes.append((x1, y1, x2, y2, conf_val, cls_id))

    # Nếu model không detect được gì, dùng heuristic đơn giản (optional)
    if len(boxes) == 0:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            ar = ww / (hh + 1e-6)
            area = ww * hh
            if 2000 < area < w * h * 0.5 and 2.0 < ar < 6.5:
                candidates.append((x, y, x + ww, y + hh, 0.5, None))
        boxes = candidates

    # Vẫn không có box ⇒ OCR luôn cả ảnh
    if len(boxes) == 0:
        boxes = [(0, 0, w - 1, h - 1, 1.0, None)]

    seen_texts = set()  # tránh trùng biển giữa các box

    base_stem = Path(image_path).stem if image_path else "image"

    # Crop từng box và OCR
    for idx, (x1, y1, x2, y2, conf_val, cls_id) in enumerate(boxes):
        crop = img_bgr[y1:y2, x1:x2].copy()
        Hc, Wc = crop.shape[:2]

        # phóng to crop nếu quá nhỏ
        if min(Hc, Wc) < 40:
            scale = max(1, int(80 / max(Hc, Wc)))
            crop = cv2.resize(
                crop,
                (Wc * scale, Hc * scale),
                interpolation=cv2.INTER_CUBIC,
            )

        # tăng tương phản nhẹ (CLAHE)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        crop_enh = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # OCR nhiều góc (0° & 180°) → chuỗi raw_text tốt nhất
        raw_text, best_angle = ocr_plate_multi_orient(
            ocr, crop_enh, use_paddle=use_paddle
        )

        # Tách thành các biển chuẩn hoá
        plate_list = extract_plate_strings(raw_text)
        if not plate_list and raw_text:
            # Không match pattern thì giữ nguyên 1 chuỗi
            plate_list = [raw_text]

        # Lưu crop
        crop_name = f"{base_stem}{name_suffix}_crop_{idx}.jpg"
        crop_path = out_dir / crop_name
        cv2.imwrite(str(crop_path), crop_enh)

        # Mỗi plate là 1 detection riêng
        for plate_text in plate_list:
            if not plate_text:
                continue
            if plate_text in seen_texts:
                continue
            seen_texts.add(plate_text)

            detections.append(
                {
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": float(conf_val) if conf_val is not None else None,
                    "class_id": int(cls_id) if cls_id is not None else None,
                    "crop_path": str(crop_path),
                    "raw_text": raw_text,
                    "text": plate_text,
                    "angle_used": int(best_angle),
                }
            )

    return detections


# ---- Utility: process folder ----
def process_source(
    model_path: str,
    source: str,
    out_dir: str = OUT_DIR_DEFAULT,
    conf: float = CONF_DEFAULT,
    ocr_lang: str = OCR_LANG_DEFAULT,
    use_paddle: bool = USE_PADDLE_DEFAULT,
):
    """
    source: single image path hoặc folder.
    Ở đây có thêm bước:
      - Đọc ảnh gốc
      - Cắt đôi + xoay trái/phải
      - Chạy detect_and_ocr cho từng nửa.
    """
    p = Path(source)
    images = []
    if p.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.png"):
            images.extend(sorted(p.glob(ext)))
    else:
        images = [p]

    results_all = {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(images, desc="Processing images"):
        try:
            # Đọc ảnh gốc
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise FileNotFoundError(f"Không thể đọc ảnh: {img_path}")

            base_stem = Path(img_path).stem

            # CẮT ĐÔI + XOAY
            sub_images = split_and_rotate_two_plates(img_bgr, base_stem)

            results_per_image = {}

            for suffix, sub_img in sub_images:
                detections = detect_and_ocr(
                    model_path=str(model_path),
                    image_path=str(img_path),   # chỉ để đặt tên crop
                    image_bgr=sub_img,
                    conf=conf,
                    ocr_lang=ocr_lang,
                    use_paddle=use_paddle,
                    name_suffix=f"_{suffix}",
                )
                results_per_image[suffix] = detections

                # In nhanh biển số đọc được
                if detections:
                    texts = [d["text"] for d in detections if d.get("text")]
                    if texts:
                        print(f"\n📌 {img_path} [{suffix}]: {texts}")
                    else:
                        print(f"\n📌 {img_path} [{suffix}]: (không đọc được text)")
                else:
                    print(f"\n📌 {img_path} [{suffix}]: (không detect được biển số)")

            results_all[str(img_path)] = results_per_image

        except Exception as ex:
            print(f"❗ Lỗi xử lý {img_path}: {ex}")
            results_all[str(img_path)] = {"error": str(ex)}

    # save json results
    json_path = Path(out_dir) / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Hoàn tất. Kết quả lưu ở: {json_path}")
    return results_all


# ---- main: tất cả dùng mặc định ----
def main():
    print("===== Biển số xe: YOLOv8 + OCR (chạy mặc định) =====")
    print(f"🔹 Model:   {MODEL_DEFAULT}")
    print(f"🔹 Source:  {SOURCE_DEFAULT}")
    print(f"🔹 Output:  {OUT_DIR_DEFAULT}")
    print(f"🔹 Conf:    {CONF_DEFAULT}")
    print(f"🔹 OCR lang:{OCR_LANG_DEFAULT}")
    print("==============================================\n")

    process_source(
        model_path=MODEL_DEFAULT,
        source=SOURCE_DEFAULT,
        out_dir=OUT_DIR_DEFAULT,
        conf=CONF_DEFAULT,
        ocr_lang=OCR_LANG_DEFAULT,
        use_paddle=USE_PADDLE_DEFAULT,
    )


if __name__ == "__main__":
    main()
