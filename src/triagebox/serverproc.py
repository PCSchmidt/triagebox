"""Start and stop a local llama-server process for one model configuration."""

from __future__ import annotations

import subprocess
import time
import urllib.request
from pathlib import Path


class ServerStartError(RuntimeError):
    pass


class ServerProc:
    """Context manager: ``with ServerProc(exe, model) as s:`` gives a healthy server on ``s.url``.

    The first shard of a split GGUF is enough; llama.cpp finds the rest.
    """

    def __init__(
        self,
        exe: Path,
        model: Path,
        port: int = 8080,
        ctx: int = 4096,
        gpu_layers: int = 99,
        seed: int = 0,
        log: Path | None = None,
        start_timeout_s: float = 240.0,
        extra_args: tuple[str, ...] = (),
    ):
        self.exe, self.model, self.port = Path(exe), Path(model), port
        self.ctx, self.gpu_layers, self.seed = ctx, gpu_layers, seed
        self.log, self.start_timeout_s = log, start_timeout_s
        self.extra_args = tuple(extra_args)
        self._proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        return [
            str(self.exe),
            "-m", str(self.model),
            "-ngl", str(self.gpu_layers),
            "-c", str(self.ctx),
            "-np", "1",
            "--seed", str(self.seed),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--no-webui",
            *self.extra_args,
        ]

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=3) as r:
                return r.status == 200
        except OSError:
            return False

    def __enter__(self) -> "ServerProc":
        if not self.model.exists():
            raise ServerStartError(f"model file not found: {self.model}")
        if self._healthy():
            raise ServerStartError(f"something is already serving on port {self.port}")
        out = open(self.log, "ab") if self.log else subprocess.DEVNULL
        self._proc = subprocess.Popen(self.command(), stdout=out, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.start_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise ServerStartError(f"llama-server exited early with code {self._proc.returncode}")
            if self._healthy():
                return self
            time.sleep(1.0)
        self.__exit__(None, None, None)
        raise ServerStartError("llama-server did not become healthy in time")

    def __exit__(self, *exc) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=10)
        self._proc = None
