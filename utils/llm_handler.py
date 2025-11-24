"""OpenRouter chat completion helper with profile-aware defaults."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import requests
from pydantic import BaseModel, Field

try:  # Streamlit is optional for local unit tests
    import streamlit as st  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    st = None

from ..modules.config import AppConfig


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 2048
    top_p: float = 0.95
    stop: Optional[List[str]] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    model: str
    content: str
    usage: Dict[str, int] = Field(default_factory=dict)


class OpenRouterClient:
    """Thin wrapper over the OpenRouter /chat/completions endpoint."""

    def __init__(self, config: AppConfig, api_key: Optional[str] = None):
        self.config = config
        self.api_key = api_key or self._resolve_api_key()
        self.base_url = config.openrouter.api_base.rstrip("/")

    def _resolve_api_key(self) -> str:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key and st is not None:
            key = st.secrets.get(  # type: ignore[attr-defined]
                "OPENROUTER_API_KEY",
                "",
            )
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY missing. "
                "Add it to Streamlit Secrets or env vars."
            )
        return key

    def _headers(self) -> Dict[str, str]:
        metadata = getattr(self.config, "metadata", {}) or {}
        referer = metadata.get("referer", "")
        title = metadata.get("app_title", "AI Content Optimizer")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": referer,
            "X-Title": title,
            "Content-Type": "application/json",
        }

    def send(self, request: LLMRequest) -> LLMResponse:
        model_id = self.config.validate_model(request.model)
        payload = request.dict()
        payload["model"] = model_id

        response = requests.post(
            url=f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            model=model_id,
            content=choice["message"]["content"],
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )

    def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> LLMResponse:
        req = LLMRequest(
            model=model or self.config.selected_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )
        return self.send(req)
