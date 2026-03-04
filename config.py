import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Tokens and keys
TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "8292116348:AAGJbnxdNzga0Fu1s7aToy00GGBtSW4unok")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "1936843947")
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "---------xxxx---")
LLM_API_URL: str = os.getenv("LLM_API_URL", "https://models.inference.ai.azure.com/chat/completions")
 
    
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
