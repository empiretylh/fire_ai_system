"""Utility helpers for Fire AI System."""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PyQt5.QtGui import QImage

import config


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, (x2 - x1) * (y2 - y1))


@dataclass
class FrameDetections:
    frame: np.ndarray
    detections: List[Detection]
    timestamp: float


@dataclass
class RiskResult:
    risk_level: str
    explanation: str
    metrics: dict


def ensure_directories() -> None:
    Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.ALERTS_DIR).mkdir(parents=True, exist_ok=True)


def now_ts() -> float:
    return time.time()


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        color = (0, 0, 255) if det.label.lower() == "fire" else (160, 160, 160)
        x1, y1, x2, y2 = map(int, det.box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated


def to_qimage(frame: np.ndarray) -> QImage:
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    return QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)


def save_snapshot(frame: np.ndarray) -> str:
    date_dir = Path(config.ALERTS_DIR) / datetime.utcnow().strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    filename = date_dir / f"alert_{int(time.time())}.jpg"
    cv2.imwrite(str(filename), frame)
    return str(filename)


def append_detection_log(timestamp: float, detections: List[Detection]) -> None:
    Path(config.DETECTIONS_CSV).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(config.DETECTIONS_CSV).exists()
    with open(config.DETECTIONS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "label", "confidence", "x1", "y1", "x2", "y2"])
        for det in detections:
            x1, y1, x2, y2 = det.box
            writer.writerow([timestamp, det.label, f"{det.confidence:.4f}", f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}"])


def append_risk_history(entry: dict) -> None:
    Path(config.RISK_HISTORY_JSON).parent.mkdir(parents=True, exist_ok=True)
    history: List[dict] = []
    if Path(config.RISK_HISTORY_JSON).exists():
        with open(config.RISK_HISTORY_JSON, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    history.append(entry)
    with open(config.RISK_HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def calc_metrics(detections: List[Detection], duration_seconds: float) -> dict:
    fire_dets = [d for d in detections if d.label.lower() == "fire"]
    smoke_dets = [d for d in detections if d.label.lower() == "smoke"]
    max_fire_area = max([d.area for d in fire_dets], default=0.0)
    metrics = {
        "fire_count": len(fire_dets),
        "smoke_count": len(smoke_dets),
        "max_fire_area": int(max_fire_area),
        "duration_seconds": int(duration_seconds),
    }
    return metrics
