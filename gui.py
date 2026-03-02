"""PyQt5 GUI for Fire AI System."""
from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QUrl
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QFileDialog,
    QComboBox,
    QCheckBox,
    QWidget,
)

import config
from utils import FrameDetections, to_qimage


class FireAIGUI(QWidget):
    # Define thread-safe signals for background threads to safely interact with UI
    sig_update_live = pyqtSignal(object)  # passing cv2 numpy frame
    sig_update_detect = pyqtSignal(object)  # passing FrameDetections
    sig_update_risk = pyqtSignal(str, str)
    sig_tele_status = pyqtSignal(str)
    sig_loading = pyqtSignal(bool, str)
    sig_play_alarm = pyqtSignal()

    def __init__(
        self,
        start_callback: Callable[[str, float, str, str, str], None],
        stop_callback: Callable[[], None],
        test_telegram_callback: Callable[[str], None],
        load_image_callback: Callable[[str], None],
    ) -> None:
        super().__init__()
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.test_telegram_callback = test_telegram_callback
        self.load_image_callback = load_image_callback
        self._build_ui()
        self._connect_signals()

    def _connect_signals(self) -> None:
        """Map thread-safe signals to their actual UI manipulation methods."""
        self.sig_update_live.connect(self.update_live_frame)
        self.sig_update_detect.connect(self.update_detect_frame)
        self.sig_update_risk.connect(self.update_risk)
        self.sig_tele_status.connect(self.set_telegram_status)
        self.sig_loading.connect(self.set_loading_state)
        self.sig_play_alarm.connect(self.play_alarm)

    def _build_ui(self) -> None:
        self.setWindowTitle(config.WINDOW_TITLE)
        self.setStyleSheet(
            """
            QWidget { background-color: #121212; color: #e0e0e0; }
            QLabel { color: #e0e0e0; }
            QPushButton { background-color: #1e88e5; color: white; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #1565c0; }
            QLineEdit { background-color: #1e1e1e; color: #e0e0e0; padding: 6px; border: 1px solid #333; }
            QSlider::groove:horizontal { height: 6px; background: #333; }
            QSlider::handle:horizontal { background: #1e88e5; width: 14px; margin: -4px 0; border-radius: 7px; }
            """
        )

        self.live_video_label = QLabel("WebCam or CCTV Input (Live)")
        self.live_video_label.setAlignment(Qt.AlignCenter)
        self.live_video_label.setMinimumHeight(480)
        self.live_video_label.setStyleSheet("background-color: #000; border: 1px solid #333;")

        self.detect_video_label = QLabel("Fire Result Output (5s Snapshot)")
        self.detect_video_label.setAlignment(Qt.AlignCenter)
        self.detect_video_label.setMinimumHeight(480)
        self.detect_video_label.setStyleSheet("background-color: #000; border: 1px solid #333;")

        self.loading_overlay = QLabel("⏳ Analyzing Risk with AI...\nPlease wait", self.detect_video_label)
        self.loading_overlay.setAlignment(Qt.AlignCenter)
        self.loading_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 180); color: white; font-size: 20px; font-weight: bold;")
        self.loading_overlay.setVisible(False)
        
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.live_video_label)
        video_layout.addWidget(self.detect_video_label)

        self.fire_label = QLabel("Fire: 0")
        self.smoke_label = QLabel("Smoke: 0")
        self.risk_label = QLabel("Risk: -")
        self.risk_label.setStyleSheet("color: #fff; font-weight: bold;")
        self.tele_status = QLabel("Telegram: idle")

        self.rtsp_input = QLineEdit(config.RTSP_DEFAULT)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(30)
        self.conf_slider.setMaximum(95)
        self.conf_slider.setValue(int(config.CONF_THRESHOLD * 100))
        self.conf_value = QLabel(f"Confidence: {config.CONF_THRESHOLD:.2f}")
        self.conf_slider.valueChanged.connect(self._on_conf_change)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("HSV filter (no api call for ai and telegram for testing)", "hsv")
        self.mode_combo.addItem("AI Model (recommended) (optimized150.pt)", "models/optimized150.pt")
        self.mode_combo.addItem("High Detection Model (fire_smoke.pt)", "models/fire_smoke.pt")

        self.prompt_input = QLineEdit("E.g., Kitchen CCTV, lots of cooking happens here.")
        self.prompt_input.setPlaceholderText("Enter context for the AI (e.g., room name, hazards)")

        self.chat_id_input = QLineEdit(config.TELEGRAM_CHAT_ID)
        self.chat_id_input.setPlaceholderText("Comma-separated Telegram Chat IDs")

        start_btn = QPushButton("Start")
        stop_btn = QPushButton("Stop")
        telegram_btn = QPushButton("Test Telegram")
        load_btn = QPushButton("Load Image")
        start_btn.clicked.connect(self._on_start)
        stop_btn.clicked.connect(self.stop_callback)
        telegram_btn.clicked.connect(self._on_test_telegram)
        load_btn.clicked.connect(self._on_load_image)

        top_controls = QHBoxLayout()
        top_controls.addWidget(QLabel("RTSP / Camera URL:"))
        top_controls.addWidget(self.rtsp_input)
        top_controls.addWidget(QLabel("Mode:"))
        top_controls.addWidget(self.mode_combo)

        self.mute_checkbox = QCheckBox("Mute Alarm")
        self.mute_checkbox.setStyleSheet("color: white; font-weight: bold; margin-left: 10px;")
        top_controls.addWidget(self.mute_checkbox)

        context_controls = QHBoxLayout()
        context_controls.addWidget(QLabel("AI Context Prompt:"))
        context_controls.addWidget(self.prompt_input)
        context_controls.addWidget(QLabel("Chat IDs:"))
        context_controls.addWidget(self.chat_id_input)

        slider_row = QHBoxLayout()
        slider_row.addWidget(self.conf_value)
        slider_row.addWidget(self.conf_slider)

        btn_row = QHBoxLayout()
        btn_row.addWidget(start_btn)
        btn_row.addWidget(stop_btn)
        btn_row.addWidget(telegram_btn)
        btn_row.addWidget(load_btn)

        status_grid = QGridLayout()
        status_grid.addWidget(self.fire_label, 0, 0)
        status_grid.addWidget(self.smoke_label, 0, 1)
        status_grid.addWidget(self.risk_label, 1, 0)
        status_grid.addWidget(self.tele_status, 1, 1)

        layout = QVBoxLayout()
        layout.addLayout(top_controls)
        layout.addLayout(context_controls)
        layout.addLayout(video_layout)
        layout.addLayout(slider_row)
        layout.addLayout(btn_row)
        layout.addLayout(status_grid)
        self.setLayout(layout)

        # Timer for potential periodic UI refresh hooks
        self.timer = QTimer()
        self.timer.start(500)

        # Alarm Player
        self.alarm_player = QMediaPlayer()
        self.alarm_player.setMedia(QMediaContent(QUrl.fromLocalFile("alarm-sound.mp3")))
        # Keep volume high and avoid resetting it constantly
        self.alarm_player.setVolume(100)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Keep the overlay exactly the same size as the underlying label
        self.loading_overlay.resize(self.detect_video_label.size())

    def _on_conf_change(self, value: int) -> None:
        conf = value / 100
        self.conf_value.setText(f"Confidence: {conf:.2f}")

    def _on_start(self) -> None:
        rtsp = self.rtsp_input.text().strip()
        conf = self.conf_slider.value() / 100
        mode = self.mode_combo.currentData()
        prompt = self.prompt_input.text().strip()
        if prompt == "E.g., Kitchen CCTV, lots of cooking happens here.":
            prompt = ""
        chat_ids = self.chat_id_input.text().strip()
        self.start_callback(rtsp, conf, mode, prompt, chat_ids)

    def _on_test_telegram(self) -> None:
        chat_ids = self.chat_id_input.text().strip()
        self.test_telegram_callback(chat_ids)

    def _on_load_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select image for testing", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.load_image_callback(file_path)

    def update_live_frame(self, frame) -> None:
        if frame is None:
            return
        image = to_qimage(frame)
        pix = QPixmap.fromImage(image).scaled(
            self.live_video_label.width(), self.live_video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.live_video_label.setPixmap(pix)

    def update_detect_frame(self, frame: FrameDetections) -> None:
        image = to_qimage(frame.frame)
        pix = QPixmap.fromImage(image).scaled(
            self.detect_video_label.width(), self.detect_video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.detect_video_label.setPixmap(pix)
        fire_count = sum(1 for d in frame.detections if d.label == "fire")
        smoke_count = sum(1 for d in frame.detections if d.label == "smoke")
        self.fire_label.setText(f"Fire: {fire_count}")
        self.smoke_label.setText(f"Smoke: {smoke_count}")

    def update_risk(self, risk_level: str, explanation: str) -> None:
        color = {
            "LOW": "#64dd17",
            "MEDIUM": "#fbc02d",
            "HIGH": "#ef6c00",
            "CRITICAL": "#d50000",
        }.get(risk_level.upper(), "#e0e0e0")
        self.risk_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.risk_label.setText(f"Risk: {risk_level} - {explanation}")

    def set_telegram_status(self, text: str) -> None:
        self.tele_status.setText(text)

    def set_loading_state(self, is_loading: bool, text: str = "⏳ Analyzing Risk with AI...\nPlease wait") -> None:
        self.loading_overlay.setText(text)
        self.loading_overlay.setVisible(is_loading)

    def play_alarm(self) -> None:
        if self.mute_checkbox.isChecked():
            return
        # Only replay if it's strictly stopped or finished so it doesn't stutter rapid-fire
        if self.alarm_player.state() != QMediaPlayer.PlayingState:
            self.alarm_player.setPosition(0)
            self.alarm_player.play()

    # --- Thread-Safe Emit Wrappers for main.py ---
    def emit_live_frame(self, frame) -> None:
        self.sig_update_live.emit(frame)

    def emit_detect_frame(self, fd: FrameDetections) -> None:
        self.sig_update_detect.emit(fd)

    def emit_risk(self, risk_level: str, explanation: str) -> None:
        self.sig_update_risk.emit(risk_level, explanation)

    def emit_tele_status(self, text: str) -> None:
        self.sig_tele_status.emit(text)

    def emit_loading(self, is_loading: bool, text: str = "⏳ Analyzing Risk with AI...\nPlease wait") -> None:
        self.sig_loading.emit(is_loading, text)

    def emit_play_alarm(self) -> None:
        self.sig_play_alarm.emit()


def launch_gui(
    start_cb: Callable[[str, float, str, str, str], None], stop_cb: Callable[[], None], test_cb: Callable[[str], None], load_cb: Callable[[str], None]
) -> QApplication:
    app = QApplication([])
    ui = FireAIGUI(start_cb, stop_cb, test_cb, load_cb)
    ui.show()
    return app, ui
