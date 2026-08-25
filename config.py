COCO_CLASS_ID_TO_NAME = {
    0: "person", 1: "bicycle", 2: "car",
    3: "motorbike", 5: "bus", 7: "truck",
}
CATEGORY_MAP = {
    "person": "human", "bicycle": "vehicle", "car": "vehicle",
    "motorbike": "vehicle", "bus": "vehicle", "truck": "vehicle",
}
TARGET_CLASS_IDS = list(COCO_CLASS_ID_TO_NAME.keys())
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.5
MODEL_WEIGHTS = "yolov8n.pt"
TRACKER_CONFIG = "bytetrack.yaml"
