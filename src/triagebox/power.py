"""GPU power and state sampling via ``nvidia-smi``, with per-window summaries.

``nvidia-smi`` reports the GPU board's power draw only, not the whole system, and
at a coarse interval. Energy figures built from it are GPU energy per answer, a
proxy for edge-device energy, not a wall-power measurement.

State (SM clock, temperature, throttle reasons) is sampled alongside power
because speed and power measured on a throttling GPU describe the throttle, not
the model: a laptop GPU here fell from 1785 MHz to 210 MHz at 94 C under sustained
generation. ``timing_trustworthy`` encodes the rule used to flag such runs.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass

# Throttle-reason bits that mean the GPU is slowed by power or heat (not merely idle
# or using application clocks).
THROTTLE_BITS = 0x04 | 0x08 | 0x20 | 0x40 | 0x80
QUERY = "power.draw,clocks.sm,clocks.max.sm,temperature.gpu,clocks_throttle_reasons.active"


@dataclass(frozen=True)
class PowerSample:
    t: float  # time.perf_counter() clock
    watts: float
    sm_mhz: int | None = None
    sm_max_mhz: int | None = None
    temp_c: int | None = None
    throttle_mask: int | None = None


def parse_power_line(line: str) -> float | None:
    """Watts from one ``nvidia-smi --format=csv,noheader,nounits`` line, or ``None``."""
    s = line.strip()
    try:
        return float(s)
    except ValueError:
        return None  # e.g. "[N/A]"


def parse_state_line(line: str, t: float) -> PowerSample | None:
    """A sample from a ``QUERY`` line: ``watts, sm, sm_max, temp, 0xmask``."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 5:
        return None
    try:
        return PowerSample(t, float(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4], 16))
    except ValueError:
        return None


def mean_between(samples: list[PowerSample], t0: float, t1: float) -> float | None:
    """Mean watts of samples in ``[t0, t1]``; if none, the latest sample before ``t1``; else ``None``."""
    inside = [s.watts for s in samples if t0 <= s.t <= t1]
    if inside:
        return sum(inside) / len(inside)
    before = [s for s in samples if s.t < t1]
    return before[-1].watts if before else None


@dataclass(frozen=True)
class GpuState:
    n_samples: int
    median_sm_mhz: float | None
    max_sm_mhz: int | None
    clock_fraction: float | None  # median SM clock / maximum SM clock
    max_temp_c: int | None
    throttle_reasons_seen: int  # OR of throttle masks in the window


def state_between(samples: list[PowerSample], t0: float, t1: float) -> GpuState:
    inside = [s for s in samples if t0 <= s.t <= t1 and s.sm_mhz is not None]
    if not inside:
        return GpuState(0, None, None, None, None, 0)
    clocks = sorted(s.sm_mhz for s in inside)
    median = clocks[len(clocks) // 2] if len(clocks) % 2 else (clocks[len(clocks) // 2 - 1] + clocks[len(clocks) // 2]) / 2
    mx = max(s.sm_max_mhz for s in inside)
    seen = 0
    for s in inside:
        seen |= s.throttle_mask or 0
    return GpuState(len(inside), median, mx, median / mx if mx else None, max(s.temp_c for s in inside), seen)


def timing_trustworthy(state: GpuState, min_clock_fraction: float = 0.9) -> bool:
    """False if the GPU was power/thermally throttled or ran well below its maximum clock."""
    if state.n_samples == 0 or state.clock_fraction is None:
        return False
    return state.clock_fraction >= min_clock_fraction and not (state.throttle_reasons_seen & THROTTLE_BITS)


class GpuPower:
    """Context manager that samples GPU power and state in a background thread."""

    def __init__(self, interval_ms: int = 100, gpu_index: int = 0):
        self.interval_ms, self.gpu_index = interval_ms, gpu_index
        self.samples: list[PowerSample] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GpuPower":
        self._proc = subprocess.Popen(
            ["nvidia-smi", f"--id={self.gpu_index}", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits", f"-lms={self.interval_ms}"],
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
            s = parse_state_line(line, time.perf_counter())
            if s is not None:
                self.samples.append(s)

    def __exit__(self, *exc) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        if self._thread:
            self._thread.join(timeout=2)

    def mean_watts(self, t0: float, t1: float) -> float | None:
        return mean_between(self.samples, t0, t1)

    def state(self, t0: float, t1: float) -> GpuState:
        return state_between(self.samples, t0, t1)


def gpu_temp_c(gpu_index: int = 0) -> int | None:
    """Current GPU temperature in Celsius from ``nvidia-smi``, or ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return int(out)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


@dataclass(frozen=True)
class CoolResult:
    reached: bool  # temperature fell to the target before the timeout
    temp_c: int | None  # last reading
    waited_s: float


def wait_for_cool(
    target_c: int,
    timeout_s: float = 900.0,
    poll_s: float = 5.0,
    read_temp=gpu_temp_c,
    sleep=time.sleep,
    clock=time.monotonic,
) -> CoolResult:
    """Block until the GPU is at or below ``target_c`` (or ``timeout_s`` passes).

    Used before each timing burst so every measurement starts from a similar
    thermal state. A failed temperature read counts as "not cool".
    """
    start = clock()
    temp = None
    while True:
        temp = read_temp()
        if temp is not None and temp <= target_c:
            return CoolResult(True, temp, clock() - start)
        if clock() - start >= timeout_s:
            return CoolResult(False, temp, clock() - start)
        sleep(poll_s)
