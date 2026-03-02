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


class CameraReader(threading.Thread):
    def __init__(self, source: str | int, stop_event: threading.Event, live_ui_cb=None):
        super().__init__(daemon=True)
        self.source = source
        self.stop_event = stop_event
        self.live_ui_cb = live_ui_cb
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[cv2.Mat] = None
        self.lock = threading.Lock()
        self.error = False

    def run(self):
        self._open_capture()
        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                self.error = True
                time.sleep(config.CAP_RECONNECT_SECONDS)
                self._open_capture()
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                self.error = True
                self._release_capture()
                continue
                
            self.error = False
            with self.lock:
                self.frame = frame.copy()
            if self.live_ui_cb and not self.stop_event.is_set():
                try:
                    self.live_ui_cb(self.frame)
                except RuntimeError:
                    # Occurs if PyQt deletes the UI label mid-cycle during application shutdown
                    pass
            # Fast frame acquisition yield
            time.sleep(0.01)
        self._release_capture()

    def get_frame(self):
        with self.lock:
            return self.frame, self.error

    def _open_capture(self) -> None:
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
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

class CaptureWorker(threading.Thread):
    """Captures frames from CameraReader, detects, analyzes, and alerts if needed."""

    def __init__(
        self,
        camera_reader: CameraReader,
        detector: FireSmokeDetector,
        analyzer: RiskAnalyzer,
        alertor: TelegramAlerter,
        ui_detect_cb,
        ui_risk_cb,
        ui_loading_cb,
        ui_play_alarm_cb,
        tele_status_cb,
        stop_event: threading.Event,
        mode: str,
        custom_prompt: str,
        chat_ids: str,
    ) -> None:
        super().__init__(daemon=True)
        self.camera_reader = camera_reader
        self.detector = detector
        self.analyzer = analyzer
        self.alertor = alertor
        self.ui_detect_cb = ui_detect_cb
        self.ui_risk_cb = ui_risk_cb
        self.ui_loading_cb = ui_loading_cb
        self.ui_play_alarm_cb = ui_play_alarm_cb
        self.tele_status_cb = tele_status_cb
        self.stop_event = stop_event
        self.mode = mode
        self.custom_prompt = custom_prompt
        self.chat_ids = chat_ids
        self.last_capture_time = 0.0

    def run(self) -> None:
        """Main loop: check frame, detect, run AI logic asynchronously."""
        while not self.stop_event.is_set():
            current_time = time.time()

            # Ignore the CAPTURE_INTERVAL delay if we only want rapid GUI testing in HSV mode
            if self.mode != "hsv" and current_time - self.last_capture_time < config.CAPTURE_INTERVAL:
                time.sleep(0.05)
                continue

            frame, error = self.camera_reader.get_frame()
            if error or frame is None:
                self.tele_status_cb("Camera error - reconnecting...")
                time.sleep(1)
                continue

            self.last_capture_time = current_time
            ts = now_ts()

            self.tele_status_cb(f"Detecting... (mode: {self.mode})")
            
            # Since loading happens inline, toggle "Loading UI" around detector
            if self.mode != "hsv" and (self.detector.model is None or self.detector.model_path != self.mode):
                self.ui_loading_cb(True, f"⏳ Loading AI Model...\n{self.mode}")
            
            detections = self.detector.detect(frame, mode=self.mode)
            
            if self.mode != "hsv":
                # Quickly turn off the model loading indicator off-thread
                self.ui_loading_cb(False)
                
            append_detection_log(ts, detections)

            annotated = draw_detections(frame, detections)
            fd = FrameDetections(frame=annotated, detections=detections, timestamp=ts)

            self.ui_detect_cb(fd)
            # In purely HSV mode without intervals, drawing the label takes ~0ms but we still shouldn't loop instantly
            if self.mode == "hsv":
                time.sleep(0.05)

            fire_count = sum(1 for d in detections if d.label == "fire")
            smoke_count = sum(1 for d in detections if d.label == "smoke")

            if fire_count > 0:
                self.ui_play_alarm_cb()

            if fire_count > 0 or smoke_count > 0:
                self.tele_status_cb("Fire/Smoke detected" + ("! Analyzing with AI..." if self.mode != "hsv" else "! Mode: HSV Testing"))
                
                metrics = calc_metrics(detections, duration_seconds=config.CAPTURE_INTERVAL)
                snapshot_path = save_snapshot(annotated)

                if self.mode != "hsv":
                    self.ui_loading_cb(True, "⏳ Analyzing Risk with AI...\nPlease wait")
                    threading.Thread(
                        target=self._run_ai_analysis,
                        args=(metrics, snapshot_path, detections, now_iso(), self.custom_prompt, self.chat_ids),
                        daemon=True
                    ).start()
                else:
                    self.ui_risk_cb("TESTING", "HSV mode enabled - AI analysis & Telegram skipped.")
                    if config.CAPTURE_INTERVAL > 0:
                        self.tele_status_cb(f"Monitoring (next capture in {config.CAPTURE_INTERVAL}s)")
            else:
                self.ui_risk_cb("LOW", "No fire or smoke detected")
                if config.CAPTURE_INTERVAL > 0:
                    self.tele_status_cb(f"Monitoring (next capture in {config.CAPTURE_INTERVAL}s)")
                else:
                    self.tele_status_cb(f"Monitoring live (mode: {self.mode})")

    def _run_ai_analysis(self, metrics, snapshot_path, detections, timestamp_iso, custom_prompt, chat_ids):
        try:
            result = self.analyzer.analyze_with_image(metrics, snapshot_path, custom_prompt=custom_prompt)
            append_risk_history({
                "timestamp": timestamp_iso,
                **result.metrics,
                "risk": result.risk_level,
                "explanation": result.explanation,
            })
            self.ui_risk_cb(result.risk_level, result.explanation)

            # AI check to prevent false positive alerts
            if result.risk_level.upper() == "LOW":
                self.tele_status_cb("AI flagged as false positive, alert suppressed.")
                return

            sent = self.alertor.send_alert_with_ai_analysis(
                snapshot_path, result.risk_level, result.explanation, timestamp_iso, detections, chat_ids=chat_ids
            )
            self.tele_status_cb("Alert sent to Telegram" if sent else "Alert failed")

        except Exception as e:
            self.tele_status_cb(f"Risk analysis failed: {e}")
            self.ui_risk_cb("ERROR", str(e))
        finally:
            self.ui_loading_cb(False)


class Controller:
    """Main controller coordinating capture, detection, and alerting."""

    def __init__(self, ui) -> None:
        self.ui = ui
        self.camera_reader: Optional[CameraReader] = None
        self.capture_worker: Optional[CaptureWorker] = None
        self.detector = FireSmokeDetector()
        self.analyzer = RiskAnalyzer()
        self.alertor = TelegramAlerter()
        self.running = False
        self.stop_event = threading.Event()

    def start(self, source: str, conf: float, mode: str = "yolo", custom_prompt: str = "", chat_ids: str = "") -> None:
        """Start the monitoring system."""
        if self.running:
            return

        self.running = True
        self.stop_event.clear()
        self.detector.conf_threshold = conf

        self.camera_reader = CameraReader(self._resolve_source(source), self.stop_event, self.ui.emit_live_frame)
        self.camera_reader.start()

        self.capture_worker = CaptureWorker(
            camera_reader=self.camera_reader,
            detector=self.detector,
            analyzer=self.analyzer,
            alertor=self.alertor,
            ui_detect_cb=self.ui.emit_detect_frame,
            ui_risk_cb=self.ui.emit_risk,
            ui_loading_cb=self.ui.emit_loading,
            ui_play_alarm_cb=self.ui.emit_play_alarm,
            tele_status_cb=self.ui.emit_tele_status,
            stop_event=self.stop_event,
            mode=mode,
            custom_prompt=custom_prompt,
            chat_ids=chat_ids,
        )
        self.capture_worker.start()
        self.ui.emit_tele_status(f"Monitoring live (mode: {mode})")

    def stop(self) -> None:
        """Stop the monitoring system."""
        self.running = False
        self.stop_event.set()
        if self.capture_worker:
            self.capture_worker.join(timeout=2.0)
        if self.camera_reader:
            self.camera_reader.join(timeout=2.0)
        self.capture_worker = None
        self.camera_reader = None
        self.ui.emit_tele_status("Stopped")

    def test_telegram(self, chat_ids: str) -> None:
        """Send a test Telegram message."""
        sent = self.alertor.send_text("Test alert from Fire AI System", chat_ids=chat_ids)
        self.ui.emit_tele_status("Telegram OK" if sent else "Telegram failed/cooldown")

    def load_image(self, image_path: str) -> None:
        """Load and analyze a single image."""
        frame = cv2.imread(image_path)
        if frame is None:
            self.ui.emit_tele_status("Failed to load image")
            return

        mode = self.ui.mode_combo.currentData()
        prompt = self.ui.prompt_input.text()
        chat_ids = self.ui.chat_id_input.text()

        if mode != "hsv" and (self.detector.model is None or self.detector.model_path != mode):
            self.ui.emit_loading(True, f"⏳ Loading AI Model...\n{mode}")
            self.ui.repaint()

        detections = self.detector.detect(frame, mode=mode)
        
        if mode != "hsv":
            self.ui.emit_loading(False)

        append_detection_log(now_ts(), detections)
        annotated = draw_detections(frame, detections)
        fd = FrameDetections(frame=annotated, detections=detections, timestamp=now_ts())
        self.ui.emit_detect_frame(fd)
        self.ui.emit_live_frame(frame)

        fire_count = sum(1 for d in detections if d.label == "fire")
        smoke_count = sum(1 for d in detections if d.label == "smoke")

        if fire_count > 0:
            self.ui.emit_play_alarm()

        if fire_count > 0 or smoke_count > 0:
            if mode != "hsv":
                self.ui.emit_loading(True, "⏳ Analyzing Risk with AI...\nPlease wait")
                metrics = calc_metrics(detections, duration_seconds=1)
                snapshot_path = save_snapshot(annotated)
                
                # Use background thread so we do not block GUI render thread
                threading.Thread(
                    target=self._run_manual_ai_analysis,
                    args=(metrics, snapshot_path, detections, prompt, chat_ids),
                    daemon=True
                ).start()
            else:
                self.ui.emit_risk("TESTING", "HSV mode enabled - AI analysis skipped.")
                self.ui.emit_tele_status("No AI processing for HSV")
        else:
            self.ui.emit_risk("LOW", "No fire or smoke detected")
            self.ui.emit_tele_status("No detection in image")

    def _run_manual_ai_analysis(self, metrics, snapshot_path, detections, prompt, chat_ids):
        try:
            # Analyze with AI using the actual image
            result = self.analyzer.analyze_with_image(metrics, snapshot_path, custom_prompt=prompt)
            append_risk_history({
                "timestamp": now_iso(),
                **result.metrics,
                "risk": result.risk_level,
                "explanation": result.explanation,
            })
            self.ui.emit_risk(result.risk_level, result.explanation)

            # Send alert with AI analysis for ANY risk level (even LOW/MEDIUM)
            sent = self.alertor.send_alert_with_ai_analysis(
                snapshot_path, result.risk_level, result.explanation, now_iso(), detections, chat_ids=chat_ids
            )
            self.ui.emit_tele_status("Alert sent" if sent else "Alert failed")
        except Exception:
            self.ui.emit_tele_status("Risk analysis failed")
        finally:
            self.ui.emit_loading(False)

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

    def _start(src: str, conf: float, mode: str, prompt: str, chat_ids: str) -> None:
        if controller:
            controller.start(src, conf, mode, prompt, chat_ids)

    def _stop() -> None:
        if controller:
            controller.stop()

    def _test(chat_ids: str) -> None:
        if controller:
            controller.test_telegram(chat_ids)

    def _load(path: str) -> None:
        if controller:
            controller.load_image(path)

    app, ui = launch_gui(_start, _stop, _test, _load)
    controller = Controller(ui)
    sys.exit(app.exec_())
