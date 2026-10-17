import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from triagebox.llamacpp import LlamaServer, TimedLLM
from triagebox.power import (
    PowerSample,
    mean_between,
    parse_power_line,
    parse_state_line,
    state_between,
    timing_trustworthy,
)


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


def test_parse_state_line():
    s = parse_state_line("92.5, 1785, 1785, 71, 0x0000000000000004", t=3.0)
    assert (s.watts, s.sm_mhz, s.sm_max_mhz, s.temp_c, s.throttle_mask) == (92.5, 1785, 1785, 71, 0x4)
    assert parse_state_line("[N/A], 1, 2, 3, 0x0", 0.0) is None
    assert parse_state_line("1, 2, 3", 0.0) is None


def state(sm, mask=0x1, temp=60, t=1.0):
    return PowerSample(t, 80.0, sm, 1785, temp, mask)


def test_full_speed_run_is_trustworthy():
    st = state_between([state(1785, t=1.0), state(1770, t=2.0), state(1785, t=3.0)], 0.0, 4.0)
    assert st.n_samples == 3 and st.clock_fraction > 0.99 and st.max_temp_c == 60
    assert timing_trustworthy(st)  # "idle" (0x1) is not a throttle


def test_low_clock_or_thermal_throttle_makes_timing_untrustworthy():
    low = state_between([state(210, mask=0x20, temp=94, t=1.0), state(210, mask=0x40, temp=94, t=2.0)], 0.0, 3.0)
    assert low.clock_fraction < 0.2 and low.throttle_reasons_seen == 0x60
    assert not timing_trustworthy(low)
    # Full clock but a power-cap reason was seen at some point: still flagged.
    capped = state_between([state(1785, t=1.0), state(1785, mask=0x4, t=2.0)], 0.0, 3.0)
    assert not timing_trustworthy(capped)


def test_no_samples_is_untrustworthy():
    assert not timing_trustworthy(state_between([], 0.0, 1.0))
    assert not timing_trustworthy(state_between([state(1785, t=9.0)], 0.0, 1.0))


def test_wait_for_cool_returns_when_the_target_is_reached():
    from triagebox.power import wait_for_cool

    temps = iter([88, 80, 70, 60, 50])
    now = [0.0]

    def sleep(s):
        now[0] += s

    r = wait_for_cool(60, timeout_s=100, poll_s=5, read_temp=lambda: next(temps), sleep=sleep, clock=lambda: now[0])
    assert r.reached and r.temp_c == 60 and r.waited_s == 15.0


def test_wait_for_cool_gives_up_at_the_timeout_and_treats_failed_reads_as_hot():
    from triagebox.power import wait_for_cool

    now = [0.0]

    def sleep(s):
        now[0] += s

    r = wait_for_cool(60, timeout_s=20, poll_s=5, read_temp=lambda: 85, sleep=sleep, clock=lambda: now[0])
    assert not r.reached and r.temp_c == 85 and r.waited_s >= 20
    r = wait_for_cool(60, timeout_s=10, poll_s=5, read_temp=lambda: None, sleep=sleep, clock=lambda: now[0])
    assert not r.reached and r.temp_c is None


def test_timing_steady_tolerates_a_power_cap_but_not_thermal_throttling():
    from triagebox.power import timing_steady

    capped = state_between([state(1100, mask=0x4, temp=72, t=1.0), state(1150, mask=0x5, temp=73, t=2.0)], 0.0, 3.0)
    assert timing_steady(capped) and not timing_trustworthy(capped)
    thermal = state_between([state(1100, mask=0x24, temp=72, t=1.0)], 0.0, 2.0)
    assert not timing_steady(thermal)
    brake = state_between([state(1700, mask=0x80, temp=60, t=1.0)], 0.0, 2.0)
    assert not timing_steady(brake)
    hot = state_between([state(1700, mask=0x1, temp=90, t=1.0)], 0.0, 2.0)
    assert not timing_steady(hot)
    assert not timing_steady(state_between([], 0.0, 1.0))


def test_energy_joules_holds_each_reading_until_the_next():
    from triagebox.power import energy_joules

    s = [PowerSample(0.0, 100.0), PowerSample(1.0, 200.0), PowerSample(2.0, 100.0)]
    # [0,1) at 100 W, [1,2) at 200 W, [2,3) at 100 W (last reading held to the end)
    assert energy_joules(s, 0.0, 3.0) == pytest.approx(100 + 200 + 100)
    assert energy_joules(s, 0.5, 1.5) == pytest.approx(50 + 100)  # clipped to the window
    assert energy_joules(s, 5.0, 6.0) == pytest.approx(100.0)  # last reading carried forward
    assert energy_joules([PowerSample(2.0, 80.0)], 1.0, 3.0) == pytest.approx(160.0)  # before first reading
    assert energy_joules([], 0.0, 1.0) is None
    assert energy_joules(s, 1.0, 1.0) is None
