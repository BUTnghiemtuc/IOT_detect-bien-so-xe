import os
import io
from datetime import datetime

import serial
from PIL import Image


# =============== CẤU HÌNH ===============
PORT = "COM3"       # ĐÚNG COM của ESP32-CAM
BAUDRATE = 115200
TIMEOUT = 5         # giây
SAVE_DIR = "captures_serial"  # thư mục lưu ảnh
# ========================================


def ensure_save_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)


def read_frame_from_serial(ser: serial.Serial) -> bytes:
    """
    Đọc 1 frame JPEG từ ESP32-CAM với giao thức:
      - Dò dòng "FRAME_LEN:<len>"
      - Sau đó đọc đúng <len> bytes nhị phân
    """
    length = None
    print("🔎 Đang chờ header FRAME_LEN từ ESP32...", flush=True)

    while True:
        line_bytes = ser.readline()
        if not line_bytes:
            print("⏱ Timeout khi đọc header.", flush=True)
            return b""

        try:
            line = line_bytes.decode("ascii", errors="ignore").strip()
        except Exception:
            continue

        if not line:
            continue

        print("📥 ESP32:", line, flush=True)

        if line.startswith("FRAME_LEN:"):
            try:
                length = int(line.split(":", 1)[1])
                print(f"✅ FRAME_LEN = {length}", flush=True)
                break
            except ValueError:
                print("⚠️ Không parse được FRAME_LEN, bỏ qua dòng này.", flush=True)
                continue

    if length is None or length <= 0:
        print("⚠️ FRAME_LEN không hợp lệ.", flush=True)
        return b""

    # Đọc đúng length bytes JPEG
    data = bytearray()
    print("📦 Đang đọc dữ liệu ảnh...", flush=True)
    while len(data) < length:
        chunk = ser.read(length - len(data))
        if not chunk:
            print(
                f"⏱ Timeout khi đọc ảnh, mới được {len(data)}/{length} bytes.",
                flush=True,
            )
            return b""
        data.extend(chunk)

    print("✅ Đọc xong 1 frame đầy đủ.", flush=True)

    # Đọc thêm phần "DONE" nếu có (không bắt buộc)
    tail = ser.readline()
    if tail:
        try:
            print("📥 ESP32:", tail.decode("ascii", errors="ignore").strip(), flush=True)
        except Exception:
            pass

    return bytes(data)


def in_anh_ben_trai(img: Image.Image) -> Image.Image:
    """Cắt và hiển thị nửa bên trái của ảnh."""
    w, h = img.size
    left = img.crop((0, 0, w // 2, h))
    left.show(title="Ảnh bên trái")
    return left


def in_anh_ben_phai(img: Image.Image) -> Image.Image:
    """Cắt và hiển thị nửa bên phải của ảnh."""
    w, h = img.size
    right = img.crop((w // 2, 0, w, h))
    right.show(title="Ảnh bên phải")
    return right


def save_images(img_full: Image.Image, img_left: Image.Image, img_right: Image.Image):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # tới ms
    full_path = os.path.join(SAVE_DIR, f"{ts}_full.jpg")
    left_path = os.path.join(SAVE_DIR, f"{ts}_left.jpg")
    right_path = os.path.join(SAVE_DIR, f"{ts}_right.jpg")

    img_full.save(full_path)
    img_left.save(left_path)
    img_right.save(right_path)

    print("💾 Đã lưu:")
    print("   -", full_path)
    print("   -", left_path)
    print("   -", right_path)


def main():
    ensure_save_dir()

    print(f"🔌 Mở cổng serial {PORT} @ {BAUDRATE}...", flush=True)
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT)
    except Exception as e:
        print("❌ Không mở được cổng serial:", e, flush=True)
        print("👉 Nhớ đóng Serial Monitor Arduino và kiểm tra lại số COM.")
        return

    print("✅ Serial ok. Đưa vật < 4cm để ESP32 chụp.\n", flush=True)
    print("Nhấn Ctrl+C để dừng.\n", flush=True)

    try:
        while True:
            frame_bytes = read_frame_from_serial(ser)
            if not frame_bytes:
                print("⚠️ Không nhận được frame hợp lệ, chờ tiếp...\n", flush=True)
                continue

            try:
                img = Image.open(io.BytesIO(frame_bytes))
                img = img.convert("RGB")
                print(f"🖼 Kích thước ảnh: {img.size}", flush=True)

                left = in_anh_ben_trai(img)
                right = in_anh_ben_phai(img)

                save_images(img, left, right)
                print("✅ Xử lý xong 1 frame.\n", flush=True)
            except Exception as e:
                print("❌ Lỗi khi decode/hiển thị ảnh:", e, flush=True)

    except KeyboardInterrupt:
        print("\n👋 Thoát chương trình (Ctrl+C).", flush=True)
    finally:
        ser.close()
        print("🔌 Đã đóng cổng serial.", flush=True)


if __name__ == "__main__":
    main()
