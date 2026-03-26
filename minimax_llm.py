"""MiniMax Cloud LLM provider for LangChain-ChatGLM-Webui.

Uses the MiniMax OpenAI-compatible API (https://api.minimax.io/v1) to provide
cloud-based LLM inference as an alternative to locally-loaded models.
"""

import os
import re
from typing import Any, Dict, List, Optional

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from openai import OpenAI


class ChatMiniMaxLLM(LLM):
    """LangChain LLM wrapper for MiniMax cloud models via OpenAI-compatible API."""

    model_name: str = "MiniMax-M2.7"
    temperature: float = 0.1
    max_token: int = 10000
    top_p: float = 0.9
    history: list = []
    api_key: str = ""
    api_base: str = "https://api.minimax.io/v1"
    client: Any = None

    def __init__(self, model_name: str = "MiniMax-M2.7", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "MINIMAX_API_KEY environment variable is required. "
                "Get your API key from https://platform.minimaxi.com/"
            )
        self._init_client()

    def _init_client(self):
        """Initialize the OpenAI client pointing to MiniMax API."""
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    @property
    def _llm_type(self) -> str:
        return "ChatMiniMaxLLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # MiniMax requires temperature in (0.0, 1.0]
        temperature = max(0.01, min(self.temperature, 1.0))

        messages = []
        # Include conversation history
        for h in self.history:
            if h and len(h) == 2:
                if h[0]:
                    messages.append({"role": "user", "content": h[0]})
                if h[1]:
                    messages.append({"role": "assistant", "content": h[1]})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_token,
            top_p=self.top_p,
        )

        result = response.choices[0].message.content or ""

        # Strip thinking tags that MiniMax M2.7 may produce
        result = re.sub(r"<think>.*?</think>\s*", "", result, flags=re.DOTALL)

        if stop is not None:
            result = enforce_stop_tokens(result, stop)

        self.history = self.history + [[None, result]]
        return result

    def load_llm(self, **kwargs):
        """No-op for cloud models (no local model to load)."""
        pass

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_token": self.max_token,
        }
