"""YOLOv8 fire and smoke detector wrapper."""
from __future__ import annotations

import threading
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

import config
from utils import Detection


class FireSmokeDetector:
    """Wraps YOLOv8 model for fire and smoke detection."""

    def __init__(self, model_path: str | None = None, conf_threshold: float | None = None) -> None:
        self.model_path = model_path or config.MODEL_PATH
        self.conf_threshold = conf_threshold if conf_threshold is not None else config.CONF_THRESHOLD
        self.model: Optional[YOLO] = None
        self.lock = threading.Lock()

    def detect(self, frame, mode: str = "yolo") -> List[Detection]:
        """Run detection on a single frame and return detections based on mode."""
        detections: List[Detection] = []

        if mode == "hsv":
            # Resize frame for performance if very large, but we'll use original for now
            blur = cv2.GaussianBlur(frame, (21, 21), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

            lower = np.array([18, 50, 50], dtype="uint8")
            upper = np.array([35, 255, 255], dtype="uint8")

            mask = cv2.inRange(hsv, lower, upper)
            no_red = cv2.countNonZero(mask)

            if int(no_red) > 15000:
                # Find the bounding box of the fire
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Get the largest contour representing the fire
                    c = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(c)
                    detections.append(Detection(label="fire", confidence=1.0, box=(float(x), float(y), float(x + w), float(y + h))))
            return detections

        with self.lock:
            # Mode acts as the model_path for YOLO models now
            if self.model is None or self.model_path != mode:
                self.model_path = mode
                self.model = YOLO(self.model_path)
            results = self.model.predict(source=frame, imgsz=640, conf=self.conf_threshold, verbose=False)
        
        if not results:
            return detections
        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return detections
        for box in boxes:
            conf = float(box.conf)
            cls_id = int(box.cls)
            label = names.get(cls_id, str(cls_id)).lower()
            if label not in {"fire", "smoke"}:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append(Detection(label=label, confidence=conf, box=(x1, y1, x2, y2)))
        return detections

    def warmup(self) -> None:
        """Run a dummy forward pass to warm the model."""
        dummy = cv2.cvtColor(cv2.imread(cv2.samples.findFile("lena.jpg")) if cv2.samples.findFile("lena.jpg") else cv2.UMat(), cv2.COLOR_BGR2RGB)
        try:
            self.detect(dummy)
        except Exception:
            # Warmup failures should not crash the app; ignore
            pass
