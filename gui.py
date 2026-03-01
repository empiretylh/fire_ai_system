"""PyQt5 GUI for Fire AI System."""
from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

import config
from utils import FrameDetections, to_qimage


class FireAIGUI(QWidget):
    def __init__(
        self,
        start_callback: Callable[[str, float], None],
        stop_callback: Callable[[], None],
        test_telegram_callback: Callable[[], None],
        load_image_callback: Callable[[str], None],
    ) -> None:
        super().__init__()
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.test_telegram_callback = test_telegram_callback
        self.load_image_callback = load_image_callback
        self._build_ui()

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

        self.video_label = QLabel("Video feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(360)
        self.video_label.setStyleSheet("background-color: #000; border: 1px solid #333;")

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

        start_btn = QPushButton("Start")
        stop_btn = QPushButton("Stop")
        telegram_btn = QPushButton("Test Telegram")
        load_btn = QPushButton("Load Image")
        start_btn.clicked.connect(self._on_start)
        stop_btn.clicked.connect(self.stop_callback)
        telegram_btn.clicked.connect(self.test_telegram_callback)
        load_btn.clicked.connect(self._on_load_image)

        top_controls = QHBoxLayout()
        top_controls.addWidget(QLabel("RTSP / Camera URL:"))
        top_controls.addWidget(self.rtsp_input)

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
        layout.addWidget(self.video_label)
        layout.addLayout(slider_row)
        layout.addLayout(btn_row)
        layout.addLayout(status_grid)
        self.setLayout(layout)

        # Timer for potential periodic UI refresh hooks
        self.timer = QTimer()
        self.timer.start(500)

    def _on_conf_change(self, value: int) -> None:
        conf = value / 100
        self.conf_value.setText(f"Confidence: {conf:.2f}")

    def _on_start(self) -> None:
        rtsp = self.rtsp_input.text().strip()
        conf = self.conf_slider.value() / 100
        self.start_callback(rtsp, conf)

    def _on_load_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select image for testing", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.load_image_callback(file_path)

    def update_frame(self, frame: FrameDetections) -> None:
        image = to_qimage(frame.frame)
        pix = QPixmap.fromImage(image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)
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


def launch_gui(
    start_cb: Callable[[str, float], None], stop_cb: Callable[[], None], test_cb: Callable[[], None], load_cb: Callable[[str], None]
) -> QApplication:
    app = QApplication([])
    ui = FireAIGUI(start_cb, stop_cb, test_cb, load_cb)
    ui.show()
    return app, ui
