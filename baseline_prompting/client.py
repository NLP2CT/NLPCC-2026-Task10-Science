"""OpenAI-compatible chat client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from baseline_prompting.errors import BaselineError


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0) -> None:
        if not base_url:
            raise BaselineError("base URL is required; pass --base-url or set OPENAI_BASE_URL")
        if not api_key:
            raise BaselineError("API key is required; pass --api-key or set OPENAI_API_KEY")
        self.url = normalize_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise BaselineError(f"API HTTP {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise BaselineError(f"API request failed: {exc}") from exc
        try:
            obj = json.loads(body)
            content = obj["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise BaselineError(f"unexpected API response: {body[:1000]}") from exc
        if not isinstance(content, str):
            raise BaselineError("API response message content is not a string")
        return content
