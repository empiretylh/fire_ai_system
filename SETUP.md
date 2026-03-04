# Project Setup Guide

Follow these steps to set up and run the Fire AI System.

## Prerequisites

- Python 3.8 or higher
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/) (with CUDA for GPU acceleration if available)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/fire-detection-system.git
   cd fire-detection-system/fire_ai_system
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # OR if requirements.txt is not yet available:
   pip install opencv-python ultralytics python-dotenv python-telegram-bot numpy pandas Pillow requests
   ```

## Model Download

The system requires a YOLOv8 model file (`.pt`). You can download pre-trained fire/smoke detection models from these locations:

- **Primary Model (Optimized):** [optimized150.pt](https://huggingface.co/SHOU-ISD/fire-and-smoke/resolve/main/yolov8n_1.pt) (Download and rename to `optimized150.pt`)
- **Alternative Model:** [fire_smoke.pt](https://github.com/luminous0219/fire-and-smoke-detection-yolov8/raw/main/best.pt) (Download and rename to `fire_smoke.pt`)

Place the downloaded `.pt` files in the `models/` directory.

## Configuration

1. **Environment Variables:**
   Create a `.env` file in the project root with the following content:
   ```env
   # Telegram Configuration
   TELEGRAM_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id

   # LLM API Configuration (for advanced analysis)
   LLM_API_KEY=your_ghp_key
   LLM_API_URL=https://models.inference.ai.azure.com/chat/completions
   ```

2. **System Settings:**
   Adjust thresholds and paths in `config.py` if necessary.

## Running the System

- **Start the Monitor (GUI):**
  ```bash
  python gui.py
  ```

- **Run Orchestrator (Headless):**
  ```bash
  python main.py
  ```

- **Run with Specific Source:**
  ```bash
  python main.py --source 0  # Webcam
  python main.py --source path/to/video.mp4
  ```
