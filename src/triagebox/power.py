"""GPU board-power sampling via ``nvidia-smi`` and per-window averaging.

``nvidia-smi`` reports the GPU board's power draw only, not the whole system, and
at a coarse interval. Energy figures built from it are GPU energy per answer, a
proxy for edge-device energy, not a wall-power measurement.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class PowerSample:
    t: float  # time.perf_counter() clock
    watts: float


def parse_power_line(line: str) -> float | None:
    """Watts from one ``nvidia-smi --format=csv,noheader,nounits`` line, or ``None``."""
    s = line.strip()
    try:
        return float(s)
    except ValueError:
        return None  # e.g. "[N/A]"


def mean_between(samples: list[PowerSample], t0: float, t1: float) -> float | None:
    """Mean watts of samples in ``[t0, t1]``; if none, the latest sample before ``t1``; else ``None``."""
    inside = [s.watts for s in samples if t0 <= s.t <= t1]
    if inside:
        return sum(inside) / len(inside)
    before = [s for s in samples if s.t < t1]
    return before[-1].watts if before else None


class GpuPower:
    """Context manager that samples GPU power in a background thread."""

    def __init__(self, interval_ms: int = 100, gpu_index: int = 0):
        self.interval_ms, self.gpu_index = interval_ms, gpu_index
        self.samples: list[PowerSample] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuPower":
        self._proc = subprocess.Popen(
            [
                "nvidia-smi",
                f"--id={self.gpu_index}",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
                f"-lms={self.interval_ms}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        time.sleep(0.5)  # let the first samples arrive
        return self

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            w = parse_power_line(line)
            if w is not None:
                self.samples.append(PowerSample(time.perf_counter(), w))

    def __exit__(self, *exc) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        if self._thread:
            self._thread.join(timeout=2)

    def mean_watts(self, t0: float, t1: float) -> float | None:
        return mean_between(self.samples, t0, t1)
