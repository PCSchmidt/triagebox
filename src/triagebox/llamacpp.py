"""Client for a local llama.cpp ``llama-server`` (OpenAI-compatible chat endpoint).

Standard library only. ``generate`` returns the full reply; ``stream`` yields the
text pieces as they arrive, which is what time-to-first-token needs. llama.cpp
sends one SSE event per generated token, so the number of pieces approximates the
number of tokens (a piece can be empty for special tokens, which are skipped).
"""

from __future__ import annotations

import json
import urllib.request
from typing import Iterator

from .bench import GenTiming, measure_generation


class LlamaServer:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str = "local",
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int = 0,
        timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model, self.max_tokens, self.temperature, self.seed, self.timeout = model, max_tokens, temperature, seed, timeout

    def _request(self, prompt: str, stream: bool) -> urllib.request.Request:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stream": stream,
        }
        return urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def generate(self, prompt: str) -> str:
        with urllib.request.urlopen(self._request(prompt, False), timeout=self.timeout) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"] or ""

    def stream(self, prompt: str) -> Iterator[str]:
        with urllib.request.urlopen(self._request(prompt, True), timeout=self.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    return
                choices = json.loads(data).get("choices") or []
                piece = (choices[0].get("delta") or {}).get("content") if choices else None
                if piece:
                    yield piece

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as resp:
                return resp.status == 200
        except OSError:
            return False


class TimedLLM:
    """Wraps a streaming server as an ``LLM`` and records timing for every call."""

    def __init__(self, server: LlamaServer, clock=None):
        self.server = server
        self.timings: list[tuple[float, float, GenTiming]] = []  # (start, end, timing)
        self._clock = clock

    def generate(self, prompt: str) -> str:
        import time

        clock = self._clock or time.perf_counter
        pieces: list[str] = []

        def tap() -> Iterator[str]:
            for p in self.server.stream(prompt):
                pieces.append(p)
                yield p

        start = clock()
        timing = measure_generation(tap(), clock=clock)
        self.timings.append((start, clock(), timing))
        return "".join(pieces)
