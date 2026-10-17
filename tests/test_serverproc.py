from pathlib import Path

import pytest

from triagebox.serverproc import ServerProc, ServerStartError


def test_command_line():
    cmd = ServerProc(Path("llama-server.exe"), Path("m.gguf"), port=8123, ctx=2048, gpu_layers=40, seed=3).command()
    assert cmd[0] == "llama-server.exe"
    assert cmd[cmd.index("-m") + 1] == "m.gguf"
    assert cmd[cmd.index("-ngl") + 1] == "40" and cmd[cmd.index("-c") + 1] == "2048"
    assert cmd[cmd.index("--port") + 1] == "8123" and cmd[cmd.index("--seed") + 1] == "3"
    assert "-np" in cmd and "--no-webui" in cmd and cmd[cmd.index("--host") + 1] == "127.0.0.1"


def test_missing_model_is_reported_before_anything_starts(tmp_path):
    with pytest.raises(ServerStartError, match="model file not found"):
        with ServerProc(tmp_path / "no-such.exe", tmp_path / "no-such.gguf"):
            pass
