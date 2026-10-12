"""
Direct Gemini Baseline Answering Generator.

Dedicated baseline implementation for direct LLM generation without intent routing:
User Query -> Gemini (Direct Answer Generation) -> Final Response

Strictly follows project security and experimental constraints:
- Receives ONLY the raw user query.
- Does NOT receive the intent label.
- Does NOT receive the reference ground truth.
- Does NOT use specialized handlers or tools.
- Never prints, logs, copies, or exposes API keys or secrets.
- Accurately tracks latency, prompt tokens, completion tokens, and API cost.
- Uses rate-limit pacing (>= 4.1s) and exponential backoff to respect RPM quotas.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

MODEL_PRICING = {
    "gemini-3.1-flash-lite": {"input_per_million": 0.075, "output_per_million": 0.30},
}

SYSTEM_INSTRUCTION = "You are a helpful and precise assistant. Answer the user prompt directly and accurately according to any instructions provided."

class DirectGeminiBaseline:
    def __init__(self, model_name: str = "gemini-3.1-flash-lite", temperature: float = 0.0, max_tokens: int = 512):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}" if self.api_key else None
        self.last_request_time = 0.0
        self.min_request_interval = 4.1  # 1 request / 4.1s = 14.6 RPM (under 15 RPM limit)

    def _rate_limit_wait(self):
        """Enforces minimum interval between consecutive requests to respect RPM quotas."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def generate_answer(self, query: str) -> Dict[str, Any]:
        """
        Submits the raw query directly to Gemini and returns the direct answer and resource metadata.
        """
        if not self.url or not self.api_key:
            return {
                "query": query,
                "status": "error",
                "error": "Gemini API key is not configured",
                "response_text": "[API Error] Gemini API key not found",
                "latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0
            }

        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_INSTRUCTION}]
            },
            "contents": [
                {"parts": [{"text": query}]}
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }

        max_retries = 8
        for attempt in range(max_retries):
            self._rate_limit_wait()
            start_time = time.perf_counter()
            try:
                res = requests.post(self.url, json=payload, timeout=25)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        usage = data.get("usageMetadata", {})
                        prompt_tokens = usage.get("promptTokenCount", 0)
                        completion_tokens = usage.get("candidatesTokenCount", 0)
                        total_tokens = usage.get("totalTokenCount", prompt_tokens + completion_tokens)

                        rates = MODEL_PRICING.get(self.model_name, {"input_per_million": 0.075, "output_per_million": 0.30})
                        cost_usd = (prompt_tokens / 1e6) * rates["input_per_million"] + (completion_tokens / 1e6) * rates["output_per_million"]

                        return {
                            "query": query,
                            "status": "success",
                            "response_text": text.strip(),
                            "latency_ms": round(elapsed_ms, 2),
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "cost_usd": cost_usd
                        }
                    else:
                        return {
                            "query": query,
                            "status": "empty_candidate",
                            "response_text": "",
                            "latency_ms": round(elapsed_ms, 2),
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "cost_usd": 0.0
                        }

                elif res.status_code == 429:
                    # Rate limit backoff
                    wait_sec = 10.0 * (attempt + 1)
                    print(f"    [!] Gemini Baseline 429 rate limit. Backing off for {wait_sec:.1f}s...")
                    time.sleep(wait_sec)
                    continue

                else:
                    print(f"    [!] Gemini Baseline HTTP {res.status_code}: {res.text[:100]}")
                    time.sleep(3.0 * (attempt + 1))
                    continue

            except Exception as e:
                print(f"    [!] Exception during Gemini call: {e}")
                time.sleep(3.0 * (attempt + 1))
                continue

        return {
            "query": query,
            "status": "api_failure",
            "error": "Max retries exceeded",
            "response_text": "[API Error] Request failed after retries",
            "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0
        }
