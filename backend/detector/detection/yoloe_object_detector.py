from contextlib import chdir
from pathlib import Path

import numpy as np
from ultralytics import YOLOE

YOLOE_WEIGHTS_NAME = "yoloe-26s-seg.pt"
MINIMUM_DETECTION_CONFIDENCE = 0.35
COMMONLY_MISPLACED_OBJECT_NAMES = [
    "keys", "wallet", "phone", "glasses", "sunglasses", "earbuds case", "headphones", "watch", "backpack",
    "handbag", "laptop", "tablet", "water bottle", "umbrella", "remote control", "credit card", "notebook", "pen",
]


def best_detection_per_label(result) -> dict[str, dict]:
    image_height, image_width = result.orig_shape
    best_detections: dict[str, dict] = {}
    for class_index, confidence, (x1, y1, x2, y2) in zip(
        result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()
    ):
        label = result.names[int(class_index)]
        if label in best_detections and confidence <= best_detections[label]["confidence"]:
            continue
        best_detections[label] = {
            "label": label,
            "confidence": round(confidence, 2),
            "box_xywh_normalized": [
                round(x1 / image_width, 3), round(y1 / image_height, 3),
                round((x2 - x1) / image_width, 3), round((y2 - y1) / image_height, 3),
            ],
        }
    return best_detections


class YoloeObjectDetector:
    def __init__(self, model_directory: str, device: str):
        self.device = device
        Path(model_directory).mkdir(parents=True, exist_ok=True)
        with chdir(model_directory):
            self.model = YOLOE(YOLOE_WEIGHTS_NAME)
            self.model.set_classes(COMMONLY_MISPLACED_OBJECT_NAMES)

    def detect_objects_in_batch(self, rgb_frames: list[np.ndarray]) -> list[dict[str, dict]]:
        if not rgb_frames:
            return []
        bgr_frames = [np.ascontiguousarray(rgb_frame[..., ::-1]) for rgb_frame in rgb_frames]
        results = self.model.predict(
            bgr_frames, device=self.device, conf=MINIMUM_DETECTION_CONFIDENCE, verbose=False
        )
        return [best_detection_per_label(result) for result in results]
