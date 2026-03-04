# Fire Detection System

ASCII flow, quick usage, and future work for the fire detection project.

---

## ASCII Flow Diagram

```
                   +----------------+
                   |  Video Source  |  (webcam / file / RTSP)
                   +--------+-------+
                            |
                            v
                   +--------+-------+
                   |  Preprocessing |  (resize / normalize / augment)
                   +--------+-------+
                            |
                            v
                   +--------+-------+
                   |   Detector     |  (load model -> infer -> bboxes)
                   |  `detector.py` |
                   +--------+-------+
                            |
                            v
                   +--------+-------+
                   | Post-processing|  (NMS / confidence filter)
                   +--------+-------+
                            |
                            v
                   +--------+-------+
                   | Risk Analyzer  |  (compute risk score & level)
                   | `risk_analyzer.py` |
                   +--------+-------+
                            |
        +-------------------+--------------------+
        |                                        |
        v                                        v
  +-----+------+                          +------+------+
  |  Persistence|                         |  Alerts     |
  | logs/ CSV   |                         | Telegram / UI|
  | `logs/`     |                         | `telegram_alert.py`|
  +-----+------+                          +------+------+
        |                                        |
        +-------------------+--------------------+
                            |
                            v
                   +--------+-------+
                   |    UI / GUI    |  (`gui.py`) shows overlays & status
                   +----------------+

```

## Components (quick)
- `config.py` — central configuration (thresholds, model paths, creds)
- `detector.py` — model loading and inference helpers
- `risk_analyzer.py` — computes risk score & classification
- `telegram_alert.py` — sends alerts (messages + images)
- `gui.py` — optional live display with overlays
- `main.py` — orchestrator / entrypoint
- `utils.py` — IO and helper utilities

## Quick Usage

For detailed installation and configuration steps, please refer to the [SETUP.md](file:///c:/Users/thura/projects/opencv/fire-detection-system/fire_ai_system/SETUP.md) file.

```bash
python -m venv venv
venv\Scripts\activate
pip install opencv-python torch torchvision numpy pandas Pillow python-telegram-bot
```

Run the system (examples):

```bash
python main.py                 # run headless/orchestrator
python gui.py                  # run GUI monitor
python main.py --source 0      # use webcam
python main.py --source video.mp4 --model models/fire_smoke.pt
```

Outputs:
- `logs/detections.csv` — detection events
- `logs/risk_history.json` — risk timeline
- `alerts/YYYY-MM-DD/` — saved snapshots and event folders

Configuration: edit `config.py` to set thresholds, model paths, camera index, and credentials.

## Future Work / Improvements

- Model & Accuracy:
  - Retrain with more fire/smoke samples and hard negatives.
  - Per-class threshold calibration and validation curves.
- Performance & Deployment:
  - Add ONNX export and support for TensorRT or OpenVINO for edge acceleration.
  - Provide a Dockerfile for reproducible deployments and easier edge installs.
  - Support multi-camera ingestion with worker pool.
- Reliability & Observability:
  - Add unit/integration tests for detection and alerting logic.
  - Structured logging, metrics, and health checks for long-running services.
- Alerting & UX:
  - Add additional channels (email, SMS, webhook, Slack) and deduplication.
  - Build a web dashboard for live streams and historical reports.
- Data & Continuous Learning:
  - Human-in-the-loop labeling for false positives/negatives and scheduled retraining.

---

File: README.md
