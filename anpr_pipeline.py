from collections import defaultdict, Counter
from dataclasses import dataclass, field
from anpr.plate_detector import PlateDetector
from anpr.ocr_reader import OCRReader
from anpr.config import OCR_EVERY_N_FRAMES, MIN_READS_FOR_CONFIRMATION

@dataclass
class VehicleReadout:
    track_id: object
    class_name: str
    bbox: tuple
    plate_text: object = None
    plate_confirmed: bool = False
    read_history: list = field(default_factory=list)

class ANPRPipeline:
    def __init__(self, detector):
        self.vehicle_detector = detector
        self.plate_detector = PlateDetector()
        self.ocr_reader = OCRReader()
        self._frame_counters = defaultdict(int)
        self._read_history = defaultdict(list)

    def process_frame(self, frame, tracked_objects=None):
        if tracked_objects is None:
            tracked_objects = self.vehicle_detector.process_frame(frame)
        vehicles = [o for o in tracked_objects if o.category == "vehicle"]
        return [self._process_vehicle(frame, v) for v in vehicles]

    def _process_vehicle(self, frame, vehicle):
        track_id = vehicle.track_id
        x1, y1, x2, y2 = map(int, vehicle.bbox)
        crop = frame[max(y1, 0):y2, max(x1, 0):x2]
        readout = VehicleReadout(track_id, vehicle.class_name, vehicle.bbox)

        if track_id is None:
            readout.plate_text = self._read_plate_from_crop(crop) or None
            return readout

        self._frame_counters[track_id] += 1
        if self._frame_counters[track_id] % OCR_EVERY_N_FRAMES == 1:
            text = self._read_plate_from_crop(crop)
            if text:
                self._read_history[track_id].append(text)

        history = self._read_history[track_id]
        readout.read_history = list(history)
        if len(history) >= MIN_READS_FOR_CONFIRMATION:
            text, count = Counter(history).most_common(1)[0]
            readout.plate_text = text
            readout.plate_confirmed = count >= MIN_READS_FOR_CONFIRMATION
        elif history:
            readout.plate_text = history[-1]
        return readout

    def _read_plate_from_crop(self, crop):
        if crop is None or crop.size == 0:
            return ""
        boxes = self.plate_detector.detect(crop)
        if not boxes:
            return ""
        p = max(boxes, key=lambda x: x.confidence)
        x1, y1, x2, y2 = map(int, p.bbox)
        plate = crop[max(y1,0):y2, max(x1,0):x2]
        if plate.size == 0:
            return ""
        text, _ = self.ocr_reader.read_plate(plate)
        return text
