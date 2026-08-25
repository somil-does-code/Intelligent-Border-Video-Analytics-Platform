import re
import cv2
from anpr.config import (
    OCR_BACKEND, OCR_LANGUAGES, OCR_MIN_CONFIDENCE,
    PLATE_RESIZE_WIDTH, APPLY_GRAYSCALE, APPLY_CLAHE
)

def _preprocess_plate(plate_img):
    img = plate_img.copy()
    h, w = img.shape[:2]
    if w < PLATE_RESIZE_WIDTH:
        scale = PLATE_RESIZE_WIDTH / float(w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    if APPLY_GRAYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if APPLY_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img) if len(img.shape) == 2 else img
    return img

def _clean_plate_text(raw_text):
    return re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()

class OCRReader:
    def __init__(self, backend=OCR_BACKEND):
        self.backend = backend
        if backend == "easyocr":
            import easyocr
            self.reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
        elif backend == "paddleocr":
            from paddleocr import PaddleOCR
            self.reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        else:
            raise ValueError(f"Unknown OCR backend: {backend}")

    def read_plate(self, plate_img):
        processed = _preprocess_plate(plate_img)
        return self._read_easyocr(processed) if self.backend == "easyocr" else self._read_paddleocr(processed)

    def _read_easyocr(self, img):
        results = self.reader.readtext(img)
        if not results:
            return "", 0.0
        text = _clean_plate_text("".join(r[1] for r in results))
        conf = sum(r[2] for r in results) / len(results)
        return (text, conf) if text and conf >= OCR_MIN_CONFIDENCE else ("", conf)

    def _read_paddleocr(self, img):
        result = self.reader.ocr(img, cls=True)
        if not result or not result[0]:
            return "", 0.0
        lines = result[0]
        text = _clean_plate_text("".join(line[1][0] for line in lines))
        conf = sum(line[1][1] for line in lines) / len(lines)
        return (text, conf) if text and conf >= OCR_MIN_CONFIDENCE else ("", conf)
