"""Entry point for Fire AI System - captures every 5 seconds, detects, and sends AI-analyzed alerts to Telegram."""
from __future__ import annotations

import sys
import threading
import time
from typing import Optional

import cv2

import config
from detector import FireSmokeDetector
from gui import launch_gui
from risk_analyzer import RiskAnalyzer
from telegram_alert import TelegramAlerter
from utils import (
    FrameDetections,
    RiskResult,
    append_detection_log,
    append_risk_history,
    calc_metrics,
    draw_detections,
    ensure_directories,
    now_iso,
    now_ts,
    save_snapshot,
)


class CaptureWorker(threading.Thread):
    """Captures frames at fixed intervals (every 5 seconds) for detection."""

    def __init__(
        self,
        source: str | int,
        detector: FireSmokeDetector,
        analyzer: RiskAnalyzer,
        alertor: TelegramAlerter,
        ui_update_cb,
        ui_risk_cb,
        tele_status_cb,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.source = source
        self.detector = detector
        self.analyzer = analyzer
        self.alertor = alertor
        self.ui_update_cb = ui_update_cb
        self.ui_risk_cb = ui_risk_cb
        self.tele_status_cb = tele_status_cb
        self.stop_event = stop_event
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_capture_time = 0.0

    def run(self) -> None:
        """Main loop: capture every 5 seconds, detect, analyze, and alert if needed."""
        while not self.stop_event.is_set():
            current_time = time.time()

            # Check if it's time to capture (every 5 seconds)
            if current_time - self.last_capture_time < config.CAPTURE_INTERVAL:
                time.sleep(0.1)  # Small sleep to reduce CPU usage
                continue

            # Capture frame
            frame = self._capture_frame()
            if frame is None:
                self.tele_status_cb("Camera error - reconnecting...")
                time.sleep(config.CAP_RECONNECT_SECONDS)
                continue

            self.last_capture_time = current_time
            ts = now_ts()

            # Run detection
            self.tele_status_cb(f"Detecting... (interval: {config.CAPTURE_INTERVAL}s)")
            detections = self.detector.detect(frame)
            append_detection_log(ts, detections)

            # Draw detections on frame
            annotated = draw_detections(frame, detections)
            fd = FrameDetections(frame=annotated, detections=detections, timestamp=ts)

            # Update UI with detection results
            self.ui_update_cb(fd)

            # Count fire and smoke
            fire_count = sum(1 for d in detections if d.label == "fire")
            smoke_count = sum(1 for d in detections if d.label == "smoke")

            # Only analyze and alert if fire or smoke is detected
            if fire_count > 0 or smoke_count > 0:
                self.tele_status_cb("Fire/Smoke detected! Analyzing risk with AI...")

                # Calculate metrics
                metrics = calc_metrics(detections, duration_seconds=config.CAPTURE_INTERVAL)

                # Save snapshot first for AI image analysis
                snapshot_path = save_snapshot(annotated)

                # Analyze risk with AI using the actual image
                try:
                    result = self.analyzer.analyze_with_image(metrics, snapshot_path)
                    append_risk_history({
                        "timestamp": now_iso(),
                        **result.metrics,
                        "risk": result.risk_level,
                        "explanation": result.explanation,
                    })
                    self.ui_risk_cb(result.risk_level, result.explanation)

                    # Send alert with AI analysis for ANY risk level (even LOW/MEDIUM)
                    sent = self.alertor.send_alert_with_ai_analysis(
                        snapshot_path, result.risk_level, result.explanation, now_iso(), detections
                    )
                    self.tele_status_cb("Alert sent to Telegram" if sent else "Alert failed")

                except Exception as e:
                    self.tele_status_cb(f"Risk analysis failed: {e}")
                    self.ui_risk_cb("ERROR", str(e))
            else:
                self.ui_risk_cb("LOW", "No fire or smoke detected")
                self.tele_status_cb(f"Monitoring (next capture in {config.CAPTURE_INTERVAL}s)")

        self._release_capture()

    def _capture_frame(self) -> Optional[cv2.Mat]:
        """Open camera if needed and capture a single frame."""
        try:
            if self.cap is None or not self.cap.isOpened():
                self._open_capture()
                if self.cap is None:
                    return None

            ret, frame = self.cap.read()
            if not ret:
                self._release_capture()
                return None

            return frame
        except Exception:
            self._release_capture()
            return None

    def _open_capture(self) -> None:
        """Initialize video capture."""
        if self.cap is not None:
            self._release_capture()

        if isinstance(self.source, int):
            src = self.source
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        else:
            src = 0 if self.source.lower() in {"", "usb", "camera", "webcam"} else self.source
            self.cap = cv2.VideoCapture(src)

    def _release_capture(self) -> None:
        """Release video capture resources."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


class Controller:
    """Main controller coordinating capture, detection, and alerting."""

    def __init__(self, ui) -> None:
        self.ui = ui
        self.capture_worker: Optional[CaptureWorker] = None
        self.detector = FireSmokeDetector()
        self.analyzer = RiskAnalyzer()
        self.alertor = TelegramAlerter()
        self.running = False
        self.stop_event = threading.Event()

    def start(self, source: str, conf: float) -> None:
        """Start the monitoring system."""
        if self.running:
            return

        self.running = True
        self.stop_event.clear()
        self.detector.conf_threshold = conf

        self.capture_worker = CaptureWorker(
            source=self._resolve_source(source),
            detector=self.detector,
            analyzer=self.analyzer,
            alertor=self.alertor,
            ui_update_cb=self.ui.update_frame,
            ui_risk_cb=self.ui.update_risk,
            tele_status_cb=self.ui.set_telegram_status,
            stop_event=self.stop_event,
        )
        self.capture_worker.start()
        self.ui.set_telegram_status(f"Monitoring (capture every {config.CAPTURE_INTERVAL}s)")

    def stop(self) -> None:
        """Stop the monitoring system."""
        self.running = False
        self.stop_event.set()
        if self.capture_worker:
            self.capture_worker.join(timeout=2.0)
        self.capture_worker = None
        self.ui.set_telegram_status("Stopped")

    def test_telegram(self) -> None:
        """Send a test Telegram message."""
        sent = self.alertor.send_text("Test alert from Fire AI System")
        self.ui.set_telegram_status("Telegram OK" if sent else "Telegram failed/cooldown")

    def load_image(self, image_path: str) -> None:
        """Load and analyze a single image."""
        frame = cv2.imread(image_path)
        if frame is None:
            self.ui.set_telegram_status("Failed to load image")
            return

        detections = self.detector.detect(frame)
        append_detection_log(now_ts(), detections)
        annotated = draw_detections(frame, detections)
        fd = FrameDetections(frame=annotated, detections=detections, timestamp=now_ts())
        self.ui.update_frame(fd)

        fire_count = sum(1 for d in detections if d.label == "fire")
        smoke_count = sum(1 for d in detections if d.label == "smoke")

        if fire_count > 0 or smoke_count > 0:
            metrics = calc_metrics(detections, duration_seconds=1)
            snapshot_path = save_snapshot(annotated)
            try:
                # Analyze with AI using the actual image
                result = self.analyzer.analyze_with_image(metrics, snapshot_path)
                append_risk_history({
                    "timestamp": now_iso(),
                    **result.metrics,
                    "risk": result.risk_level,
                    "explanation": result.explanation,
                })
                self.ui.update_risk(result.risk_level, result.explanation)

                # Send alert with AI analysis for ANY risk level (even LOW/MEDIUM)
                sent = self.alertor.send_alert_with_ai_analysis(
                    snapshot_path, result.risk_level, result.explanation, now_iso(), detections
                )
                self.ui.set_telegram_status("Alert sent" if sent else "Alert failed")
            except Exception:
                self.ui.set_telegram_status("Risk analysis failed")
        else:
            self.ui.update_risk("LOW", "No fire or smoke detected")
            self.ui.set_telegram_status("No detection in image")

    @staticmethod
    def _resolve_source(source: str) -> str | int:
        """Resolve source string to camera index or URL."""
        normalized = (source or "").strip().lower()
        placeholder = config.RTSP_DEFAULT.lower()
        if not normalized or normalized in {"usb", "camera", "webcam", "default", "none"}:
            return config.USB_CAMERA_INDEX
        if normalized == placeholder or "username:password@ip:port" in normalized:
            return config.USB_CAMERA_INDEX
        return source


if __name__ == "__main__":
    ensure_directories()
    controller: Optional[Controller] = None

    def _start(src: str, conf: float) -> None:
        if controller:
            controller.start(src, conf)

    def _stop() -> None:
        if controller:
            controller.stop()

    def _test() -> None:
        if controller:
            controller.test_telegram()

    def _load(path: str) -> None:
        if controller:
            controller.load_image(path)

    app, ui = launch_gui(_start, _stop, _test, _load)
    controller = Controller(ui)
    sys.exit(app.exec_())
