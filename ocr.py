from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='en')
ocr.ocr('outputs/crops/oto1_crop_0.jpg', cls=True)
