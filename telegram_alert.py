"""Telegram alerting helper with AI agent analysis in Burmese."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import requests

import config
from utils import Detection


class TelegramAlerter:
    def __init__(self, token: str | None = None, chat_id: str | None = None, cooldown: int | None = None) -> None:
        self.token = token or config.TELEGRAM_TOKEN
        self.chat_id = chat_id or config.TELEGRAM_CHAT_ID
        self.cooldown = cooldown if cooldown is not None else config.ALERT_COOLDOWN
        self.last_sent_ts: float = 0.0

    @property
    def base_url(self) -> str:
        return f"https://api.telegram.org/bot{self.token}"

    def can_send(self) -> bool:
        return (time.time() - self.last_sent_ts) >= self.cooldown

    def send_alert_with_ai_analysis(
        self, image_path: str, risk_level: str, explanation: str, timestamp: str, detections: List[Detection]
    ) -> bool:
        """Send alert to Telegram with AI agent analysis of the detection in Burmese."""
        if not self.can_send():
            return False

        # Build detailed detection info for AI analysis
        detection_details = []
        for det in detections:
            x1, y1, x2, y2 = det.box
            detection_details.append(
                {
                    "label": det.label,
                    "confidence": f"{det.confidence:.2%}",
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "area": int(det.area),
                }
            )

        # AI agent generates enhanced analysis in Burmese
        ai_analysis = self._get_ai_agent_analysis(detection_details, risk_level, explanation)

        # Count fire and smoke
        fire_count = sum(1 for d in detections if d.label.lower() == "fire")
        smoke_count = sum(1 for d in detections if d.label.lower() == "smoke")

        # Burmese caption with classification
        caption = (
            f"🔥 မီးဘေး အန္တရာယ် သတိပေးချက် 🔥\n\n"
            f"📊 အချက်အလက်များ:\n"
            f"• မီးတောက်များ: {fire_count} ခု\n"
            f"• မီးခိုးများ: {smoke_count} ခု\n"
            f"• အန္တရာယ် အဆင့်: {risk_level}\n\n"
            f"🤖 AI ခွဲခြမ်းစိတ်ဖြာမှု:\n{ai_analysis}\n\n"
            f"⏰ အချိန်: {timestamp}\n"
            f"📷 စုစုပေါင်း တွေ့ရှိမှု: {len(detections)} ခု"
        )

        files = {"photo": open(Path(image_path), "rb")}
        data = {"chat_id": self.chat_id, "caption": caption}

        try:
            resp = requests.post(
                f"{self.base_url}/sendPhoto", data=data, files=files, timeout=config.TELEGRAM_TIMEOUT
            )
            resp.raise_for_status()
            self.last_sent_ts = time.time()
            return True
        except Exception:
            return False

    def _get_ai_agent_analysis(self, detections: List[dict], risk_level: str, base_explanation: str) -> str:
        """Use AI agent to generate enhanced analysis in Burmese."""
        ai_prompt = (
            f"You are a fire safety AI agent. Analyze this detection data and provide assessment in **Burmese**.\n\n"
            f"Detection Data:\n{json.dumps(detections, indent=2)}\n\n"
            f"Risk Level: {risk_level}\n"
            f"Base Analysis: {base_explanation}\n\n"
            f"**Respond in Burmese only.** Provide 2-3 sentences about:\n"
            f"- မီးဘေး အန္တရာယ်၏ အခြေအနေ (fire severity)\n"
            f"- ချက်ချင်း လုပ်ဆောင်ရမည့် အရာများ (immediate actions)\n"
            f"- ပျံ့နှံ့နိုင်ခြေ ရှိ/မရှိ (spread risk)"
        )

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a fire safety AI agent. Always respond in Burmese."},
                {"role": "user", "content": ai_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 200,
        }

        headers = {"Authorization": f"Bearer {config.LLM_API_KEY}", "Content-Type": "application/json"}

        try:
            resp = requests.post(config.LLM_API_URL, headers=headers, json=payload, timeout=config.LLM_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", base_explanation)
            return content.strip()
        except Exception:
            # Fallback Burmese message
            return f"အန္တရာယ် အဆင့်: {risk_level} - ဂရုစိုက်ပါ"

    def send_alert(self, image_path: str, risk_level: str, explanation: str, timestamp: str) -> bool:
        """Legacy method - sends alert without AI analysis."""
        if not self.can_send():
            return False
        caption = f"Risk: {risk_level}\n{explanation}\nTime: {timestamp}"
        files = {"photo": open(Path(image_path), "rb")}
        data = {"chat_id": self.chat_id, "caption": caption}
        try:
            resp = requests.post(
                f"{self.base_url}/sendPhoto", data=data, files=files, timeout=config.TELEGRAM_TIMEOUT
            )
            resp.raise_for_status()
            self.last_sent_ts = time.time()
            return True
        except Exception:
            return False

    def send_text(self, text: str) -> bool:
        if not self.can_send():
            return False
        data = {"chat_id": self.chat_id, "text": text}
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage", data=data, timeout=config.TELEGRAM_TIMEOUT
            )
            resp.raise_for_status()
            self.last_sent_ts = time.time()
            return True
        except Exception:
            return False
