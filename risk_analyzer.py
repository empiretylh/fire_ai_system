"""Risk analysis layer using LLM reasoning with image analysis in Burmese."""
from __future__ import annotations

import base64
import json
from typing import Dict

import requests

import config
from utils import RiskResult

PROMPT_TEMPLATE = (
    "You are a fire risk assessment AI. Analyze this image for fire and smoke detection.\n\n"
    "Classify risk as:\nLOW, MEDIUM, HIGH, CRITICAL.\n\n"
    "Provide:\n"
    "- Risk level (LOW/MEDIUM/HIGH/CRITICAL)\n"
    "- Explanation in Burmese (2-3 sentences about what you see in the image)\n"
    "- Severity assessment\n"
    "- Recommended immediate actions\n\n"
    "User Context/Prompt: {custom_prompt}"
)


class RiskAnalyzer:
    def __init__(self, api_key: str | None = None, api_url: str | None = None) -> None:
        self.api_key = api_key or config.LLM_API_KEY
        self.api_url = api_url or config.LLM_API_URL

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 for sending to AI."""
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return base64.b64encode(image_bytes).decode("utf-8")

    def _build_payload_with_image(self, metrics: Dict, image_base64: str, custom_prompt: str = "") -> Dict:
        prompt = PROMPT_TEMPLATE.format(
            json_data=json.dumps(metrics, indent=2),
            custom_prompt=custom_prompt if custom_prompt else "No context provided."
        )
        return {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert fire risk evaluator. Analyze images and respond in Burmese."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        }

    def analyze_with_image(self, metrics: Dict, image_path: str, custom_prompt: str = "") -> RiskResult:
        """Analyze risk with AI using both image and detection data."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Encode image to base64
        image_base64 = self._encode_image_to_base64(image_path)
        payload = self._build_payload_with_image(metrics, image_base64, custom_prompt)
        
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=config.LLM_TIMEOUT * 2)
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        risk_level, explanation = self._parse_response(content)
        return RiskResult(risk_level=risk_level, explanation=explanation, metrics=metrics)

    def analyze(self, metrics: Dict) -> RiskResult:
        """Legacy method - analyze without image."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        prompt = PROMPT_TEMPLATE.format(json_data=json.dumps(metrics, indent=2))
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an expert fire risk evaluator. Always respond in Burmese."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 250,
        }
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=config.LLM_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        risk_level, explanation = self._parse_response(content)
        return RiskResult(risk_level=risk_level, explanation=explanation, metrics=metrics)

    @staticmethod
    def _parse_response(content: str) -> tuple[str, str]:
        # Simple parsing: expect risk level keyword in text
        text = content.strip().upper()
        level = "LOW"
        for candidate in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if candidate in text:
                level = candidate
                break
        explanation = content.strip()
        return level, explanation
