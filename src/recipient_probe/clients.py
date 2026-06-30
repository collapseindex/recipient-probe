"""Minimal LLM clients for the behavioral elicitation experiment (Anthropic + OpenRouter), so the repo is
standalone. The mechanistic probe (experiments/probe_intent.py) needs none of this, it runs a local model.
Keys are read from a .env file in the repo root (ANTHROPIC_API_KEY, OPENROUTER_API_KEY)."""
from __future__ import annotations

import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "gpt4o-mini": "openai/gpt-4o-mini",
    "llama70": "meta-llama/llama-3.1-70b-instruct",
}


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


class AnthropicLLM:
    def __init__(self, model, max_tokens=1024, temperature=0.0, timeout=120):
        self.model, self.max_tokens, self.temperature, self.timeout = model, max_tokens, temperature, timeout
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY not set (.env)")

    def chat(self, messages):
        sys = "".join(m["content"] for m in messages if m["role"] == "system")
        conv = [m for m in messages if m["role"] != "system"]
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                "messages": conv}
        if sys:
            body["system"] = sys
        r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=self.timeout,
                          headers={"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []))


class OpenRouterLLM:
    def __init__(self, model, max_tokens=1024, temperature=0.0, timeout=120):
        self.model, self.max_tokens, self.temperature, self.timeout = model, max_tokens, temperature, timeout
        self.key = os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise RuntimeError("OPENROUTER_API_KEY not set (.env)")

    def chat(self, messages):
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                "temperature": self.temperature}
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.key}"})
        r.raise_for_status()
        return (r.json()["choices"][0]["message"]["content"]) or ""


def make_client(name):
    model = MODELS.get(name, name)
    return AnthropicLLM(model) if model.startswith("claude") else OpenRouterLLM(model)
