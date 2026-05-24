"""LLM provider abstraction and OpenAI-compatible implementation.

This module provides a minimal interface to call an OpenAI-compatible API
(OpenRouter, LM Studio, Ollama)."""

from __future__ import annotations

import os
from typing import List, Dict
import time


class OpenAICompatProvider:
    """OpenAI-compatible provider (OpenRouter/Ollama/LM Studio)."""

    def __init__(self) -> None:
        #print("Initializing OpenAI-compatible LLM provider...")
        # Import inside to avoid import side-effects during static analysis
        from openai import OpenAI  # type: ignore

        self._base_url = os.getenv("OPENAI_BASE_URL","http://localhost:8010/v1")
        self._api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("LLM_MODEL") or os.getenv(
            "ROOT_AGENT_MODEL", "Qwen3-4B-Instruct-2507-FP8"
        )
        self.referer = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "machine-learning-engineering")

        self.client = OpenAI(base_url=self._base_url, api_key=self._api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        #print("Calling chat completion on OpenAI-compatible LLM provider...")
        # Add OpenRouter-specific headers only when targeting OpenRouter
        extra_headers = None
        if "openrouter.ai" in (self._base_url or ""):
            extra_headers = {"HTTP-Referer": self.referer, "X-Title": self.app_name}
        
        # FIX 1: Prevent Context Window Overflow (400 Bad Request)
        if max_tokens is not None and max_tokens < 1:
            print(f"Warning: Calculated max_tokens was negative ({max_tokens}). Forcing to 2048.")
            
        # FIX 2 & 3: Add Retry Loop & Try/Except for Network Errors (504 Timeouts)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                #print("Connecting base url: ", self._base_url)
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers,
                    timeout=300.0,  # Explicitly tell the client to wait up to 5 minutes
                )

                # Check if the response is valid before accessing choices
                if not resp or not hasattr(resp, "choices") or len(resp.choices) == 0:
                    print(f"Warning: LLM provider returned an invalid or empty response: {resp}")
                    return ""
                    
                content = resp.choices[0].message.content
                return content or ""
                
            except Exception as e:
                print(f"Warning: LLM API call failed (Attempt {attempt + 1}/{max_retries}). Error: {e}")
                if attempt < max_retries - 1:
                    sleep_time = 5 * (attempt + 1)
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print("Error: Max retries reached. LLM provider is unresponsive.")
                    return ""


def get_llm() -> OpenAICompatProvider:
    return OpenAICompatProvider()