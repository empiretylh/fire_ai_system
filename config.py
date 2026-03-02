"""Configuration for Fire AI System."""
from __future__ import annotations

# Tokens and keys
TELEGRAM_TOKEN: str = "8292116348:AAGJbnxdNzga0Fu1s7aToy00GGBtSW4unok"
TELEGRAM_CHAT_ID: str = "1936843947"
LLM_API_KEY: str = "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"
LLM_API_URL: str = "https://models.inference.ai.azure.com/chat/completions"  # OpenAI-compatible endpoint
 
    
# Model and detection
MODEL_PATH: str = "models/optimized150.pt"
CONF_THRESHOLD: float = 0.6
CAPTURE_INTERVAL: int = 5  # Capture snapshot every 5 seconds
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
CAMERA_FPS: int = 15

# Alerting
ALERT_COOLDOWN: int = 60  # seconds

# Streams
RTSP_DEFAULT: str = "rtsp://username:password@ip:port/stream"
USB_CAMERA_INDEX: int = 0

# Paths
LOG_DIR: str = "logs"
ALERTS_DIR: str = "alerts"
DETECTIONS_CSV: str = f"{LOG_DIR}/detections.csv"
RISK_HISTORY_JSON: str = f"{LOG_DIR}/risk_history.json"

# GUI
WINDOW_TITLE: str = "Fire + Smoke AI Monitor"

# Threading
FRAME_QUEUE_MAX: int = 5
DETECTION_QUEUE_MAX: int = 5
RISK_QUEUE_MAX: int = 5

# Timeouts
CAP_RECONNECT_SECONDS: int = 5
LLM_TIMEOUT: int = 30  # Increased for image analysis
TELEGRAM_TIMEOUT: int = 15
