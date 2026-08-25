import os
from dataclasses import dataclass
import cv2
from anpr.config import PLATE_MODEL_WEIGHTS, PLATE_CONFIDENCE_THRESHOLD

@dataclass
class PlateBox:
    bbox: tuple
    confidence: float

class PlateDetector:
    def __init__(self, weights_path=PLATE_MODEL_WEIGHTS):
        self.model = None
        if os.path.exists(weights_path):
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
            self.mode = "yolo"
        else:
            self.mode = "classical"

    def detect(self, vehicle_crop):
        if vehicle_crop is None or vehicle_crop.size == 0:
            return []
        return self._detect_yolo(vehicle_crop) if self.mode == "yolo" else self._detect_classical(vehicle_crop)

    def _detect_yolo(self, vehicle_crop):
        results = self.model.predict(source=vehicle_crop,
                                     conf=PLATE_CONFIDENCE_THRESHOLD,
                                     verbose=False)
        if not results or results[0].boxes is None:
            return []
        return [
            PlateBox(tuple(map(float, b.xyxy[0])), float(b.conf[0]))
            for b in results[0].boxes
        ]

    def _detect_classical(self, vehicle_crop):
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(gray, 30, 200)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        plates = []
        h_img, w_img = gray.shape[:2]

        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(approx) != 4:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            if h == 0:
                continue
            ratio = w / float(h)
            if 2.0 <= ratio <= 6.0 and w > w_img * 0.15:
                plates.append(PlateBox((float(x), float(y), float(x+w), float(y+h)), 0.5))
        return plates
