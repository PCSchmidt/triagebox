"""Generation timing (time to first token, throughput), energy, and the Pareto front.

Energy per answer is average power times duration; the power figure must be
measured on the device. Latency and energy readings should be repeated and
interleaved across configurations, because a throttling GPU drifts during a run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class GenTiming:
    ttft_ms: float  # request start to first token
    total_ms: float
    n_tokens: int
    tokens_per_s: float  # decode rate: tokens after the first over the time after the first


def measure_generation(stream: Iterable[str], clock: Callable[[], float] = time.perf_counter) -> GenTiming:
    """Time a token stream. Tokens are whatever units the stream yields."""
    start = clock()
    first = None
    n = 0
    for _ in stream:
        n += 1
        if first is None:
            first = clock()
    end = clock()
    if first is None:
        raise ValueError("stream produced no tokens")
    decode_s = end - first
    rate = (n - 1) / decode_s if n > 1 and decode_s > 0 else 0.0
    return GenTiming((first - start) * 1000.0, (end - start) * 1000.0, n, rate)


def joules_per_answer(avg_watts: float, total_ms: float) -> float:
    return avg_watts * total_ms / 1000.0


@dataclass(frozen=True)
class Point:
    name: str
    quality: float  # higher is better
    joules: float  # lower is better


def pareto_front(points: list[Point]) -> list[Point]:
    """Points not dominated by another (better-or-equal on both axes, strictly better on one)."""
    front = []
    for p in points:
        dominated = any(
            q is not p
            and q.quality >= p.quality
            and q.joules <= p.joules
            and (q.quality > p.quality or q.joules < p.joules)
            for q in points
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p.joules)
