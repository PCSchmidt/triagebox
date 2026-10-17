import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from triagebox.llamacpp import LlamaServer, TimedLLM
from triagebox.power import PowerSample, mean_between, parse_power_line


class Handler(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen.append(body)
        if not body["stream"]:
            payload = json.dumps({"choices": [{"message": {"content": "ANSWER: hello [d#0]"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},  # no content: skipped
            {"choices": [{"delta": {"content": "ANSWER:"}}]},
            {"choices": [{"delta": {"content": " hi"}}]},
            {"choices": [{"delta": {"content": ""}}]},
            {"choices": []},
            {"choices": [{"delta": {"content": " [d#0]"}}]},
        ]
        for e in events:
            self.wfile.write(f"data: {json.dumps(e)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture()
def server():
    Handler.seen = []
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield LlamaServer(f"http://127.0.0.1:{httpd.server_address[1]}", max_tokens=64, seed=7)
    httpd.shutdown()


def test_generate_returns_the_message_and_sends_deterministic_settings(server):
    assert server.generate("question") == "ANSWER: hello [d#0]"
    body = Handler.seen[-1]
    assert body["temperature"] == 0.0 and body["seed"] == 7 and body["max_tokens"] == 64
    assert body["stream"] is False and body["messages"][0]["content"] == "question"


def test_stream_yields_only_content_pieces(server):
    assert list(server.stream("q")) == ["ANSWER:", " hi", " [d#0]"]
    assert Handler.seen[-1]["stream"] is True


def test_health(server):
    assert server.health() is True
    assert LlamaServer("http://127.0.0.1:9").health() is False  # nothing listening


def test_timed_llm_returns_text_and_records_timing(server):
    # clock reads: TimedLLM start, timer start, first token, timer end, TimedLLM end
    ticks = iter([0.0, 0.0, 0.2, 1.0, 1.0])
    llm = TimedLLM(server, clock=lambda: next(ticks))
    assert llm.generate("q") == "ANSWER: hi [d#0]"
    (start, end, timing), = llm.timings
    assert timing.n_tokens == 3 and timing.ttft_ms == pytest.approx(200.0)


def test_parse_power_line():
    assert parse_power_line(" 92.50\n") == 92.5
    assert parse_power_line("[N/A]") is None
    assert parse_power_line("") is None


def test_mean_between_uses_window_then_falls_back_to_latest_before():
    s = [PowerSample(1.0, 10.0), PowerSample(2.0, 20.0), PowerSample(3.0, 30.0), PowerSample(9.0, 90.0)]
    assert mean_between(s, 1.5, 3.5) == pytest.approx(25.0)
    assert mean_between(s, 4.0, 5.0) == 30.0  # nothing inside: latest before the end
    assert mean_between(s, 0.0, 0.5) is None  # nothing at or before
    assert mean_between([], 0.0, 1.0) is None
