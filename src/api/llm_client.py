import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class LLMResult:
    ok: bool
    data: Optional[Dict[str, Any]]
    raw_text: Optional[str]
    error: Optional[str]
    latency_ms: int


def _clamp_json_object(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        # Sometimes models wrap JSON in code fences — strip and retry.
        s = raw.strip()
        if s.startswith("```"):
            # remove surrounding backticks and optional "json" hint
            s = s.strip("`").lstrip()
            if s.lower().startswith("json"):
                s = s[4:]
        s = s.strip()
        try:
            return json.loads(s)
        except Exception:
            return {}


class LLMClient:
    def __init__(self) -> None:
        self.provider: Optional[str] = None
        self.api_key: Optional[str] = (os.getenv("GENAI_API_KEY") or "").strip() or None
        if self.api_key:
            self.provider = "purdue_genai"
            self._genai_url = os.getenv(
                "GENAI_API_URL",
                "https://genai.rcac.purdue.edu/api/chat/completions",
            )
            self._genai_model = os.getenv("GENAI_MODEL", "llama3.1:latest")

    def is_available(self) -> bool:
        return bool(self.provider) and bool(self.api_key)

    def ask_json(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int = 800,
        temperature: float = 0.0,
    ) -> LLMResult:
        if not self.is_available():
            return LLMResult(False, None, None, "No LLM provider configured", 0)

        start = time.time()
        try:
            raw = ""
            if self.provider == "purdue_genai":
                raw = self._call_purdue_genai(system, prompt, max_tokens, temperature)
            else:
                raise RuntimeError("Unsupported provider: {}".format(self.provider))

            parsed = _clamp_json_object(raw)
            return LLMResult(True, parsed, raw, None, int((time.time() - start) * 1000))
        except Exception as e:
            return LLMResult(False, None, None, str(e), int((time.time() - start) * 1000))

    # ---- providers ----

    def _call_purdue_genai(
        self, system: str, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """
        Calls Purdue GenAI OpenAI-compatible /chat/completions.
        We keep stream=False to simplify parsing.
        """
        api_key = os.environ["GENAI_API_KEY"]
        url = self._genai_url
        model = self._genai_model

        system_msg = (
            "You are a strict JSON generator. "
            "Only output a single valid JSON object, no prose."
            "\n\n" + system
        )
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        headers = {
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # OpenAI-style shape:
        # choices[0].message.content is the text
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return msg.get("content") or ""
