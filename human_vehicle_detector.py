from dataclasses import dataclass
from typing import List, Optional
import cv2
from ultralytics import YOLO
from detection.config import (
    COCO_CLASS_ID_TO_NAME, CATEGORY_MAP, TARGET_CLASS_IDS,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MODEL_WEIGHTS, TRACKER_CONFIG
)

@dataclass
class TrackedObject:
    track_id: Optional[int]
    class_name: str
    category: str
    confidence: float
    bbox: tuple

class HumanVehicleDetector:
    def __init__(self, weights=MODEL_WEIGHTS, confidence=CONFIDENCE_THRESHOLD,
                 iou=IOU_THRESHOLD, tracker=TRACKER_CONFIG):
        self.model = YOLO(weights)
        self.confidence = confidence
        self.iou = iou
        self.tracker = tracker

    def process_frame(self, frame) -> List[TrackedObject]:
        results = self.model.track(
            source=frame, classes=TARGET_CLASS_IDS, conf=self.confidence,
            iou=self.iou, tracker=self.tracker, persist=True, verbose=False
        )
        objects = []
        if not results or results[0].boxes is None:
            return objects

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = COCO_CLASS_ID_TO_NAME.get(class_id, "unknown")
            category = CATEGORY_MAP.get(class_name, "unknown")
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else None
            objects.append(TrackedObject(
                track_id, class_name, category, confidence, (x1, y1, x2, y2)
            ))
        return objects

    def annotate_frame(self, frame, tracked_objects):
        annotated = frame.copy()
        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj.bbox)
            color = (0, 255, 0) if obj.category == "human" else (255, 128, 0)
            label = f"{obj.class_name} #{obj.track_id} {obj.confidence:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return annotated
