
from dataclasses import dataclass
from typing import List, Optional

import cv2
from ultralytics import YOLO

from detection.config import (
    COCO_CLASS_ID_TO_NAME,
    CATEGORY_MAP,
    TARGET_CLASS_IDS,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    MODEL_WEIGHTS,
)


@dataclass
class TrackedObject:
    track_id: Optional[int]
    class_name: str
    category: str
    confidence: float
    bbox: tuple


class HumanVehicleDetector:

    def __init__(
        self,
        weights=MODEL_WEIGHTS,
        confidence=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
    ):
        print(f"[YOLO] Loading model: {weights}")

        self.model = YOLO(weights)

        self.confidence = confidence
        self.iou = iou

        print("[YOLO] Model loaded successfully")
        print(f"[YOLO] Target classes: {TARGET_CLASS_IDS}")
        print(f"[YOLO] Confidence threshold: {self.confidence}")

    def process_frame(self, frame) -> List[TrackedObject]:

        results = self.model(
            frame,
            classes=TARGET_CLASS_IDS,
            conf=self.confidence,
            iou=self.iou,
            verbose=False
        )

        objects = []

        if not results:
            return objects

        result = results[0]

        if result.boxes is None:
            return objects

        for box in result.boxes:

            class_id = int(box.cls[0])

            class_name = COCO_CLASS_ID_TO_NAME.get(
                class_id,
                "unknown"
            )

            category = CATEGORY_MAP.get(
                class_name,
                "unknown"
            )

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                float,
                box.xyxy[0]
            )

            objects.append(
                TrackedObject(
                    track_id=None,
                    class_name=class_name,
                    category=category,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2)
                )
            )

        return objects

    def annotate_frame(self, frame, tracked_objects):

        annotated = frame.copy()

        for obj in tracked_objects:

            x1, y1, x2, y2 = map(
                int,
                obj.bbox
            )

            if obj.category == "human":
                color = (0, 255, 0)
            else:
                color = (255, 128, 0)

            label = f"{obj.class_name} {obj.confidence:.0%}"

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label_width = max(len(label) * 9, 100)

            cv2.rectangle(
                annotated,
                (x1, max(y1 - 25, 0)),
                (x1 + label_width, y1),
                color,
                -1
            )

            cv2.putText(
                annotated,
                label,
                (x1 + 5, max(y1 - 7, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return annotated


