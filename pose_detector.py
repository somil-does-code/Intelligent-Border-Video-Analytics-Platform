import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class SuspiciousPoseDetector:
    def __init__(self, task_path="pose/pose_landmarker.task"):
        self.detector = None
        if not os.path.exists(task_path):
            print(f"[WARN] Pose model missing: {task_path}. Pose alerts disabled.")
            return

        base_options = python.BaseOptions(model_asset_path=task_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def process_frame(self, frame):
        if self.detector is None:
            return False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(image)

        if not result.pose_landmarks:
            return False

        lm = result.pose_landmarks[0]
        return lm[15].y < lm[11].y and lm[16].y < lm[12].y

    def close(self):
        if self.detector is not None:
            self.detector.close()
