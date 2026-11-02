import asyncio
import base64
import json
import threading
import time

import numpy as np
import pytest

from asr_dashboard.config.schema import AppConfig
from asr_dashboard.realtime.protocols.vllm_realtime import SentenceSplitter
from asr_dashboard.realtime.session import RealtimeSession, SessionCallbacks, SessionSettings
from asr_dashboard.realtime.validate import validate_async

websockets = pytest.importorskip("websockets")


class FakeVllmServer:
    """Minimal vLLM /v1/realtime: one delta '句子N。' per 0.5 s of audio; done on final commit."""

    def __init__(self, key=None, model="Qwen/Qwen3-ASR-0.6B", drop_after=None, plain_http=False):
        self.key, self.model, self.drop_after, self.plain_http = key, model, drop_after, plain_http
        self.connections = 0
        self.total_samples = 0
        self.port = None
        self._ready = threading.Event()

    async def handler(self, ws):
        self.connections += 1
        conn_no = self.connections
        await ws.send(json.dumps({"type": "session.created", "id": "s"}))
        validated = False
        samples = 0
        text = ""
        appends = 0
        async for raw in ws:
            msg = json.loads(raw)
            t = msg["type"]
            if t == "session.update":
                if msg.get("model") != self.model:
                    await ws.send(json.dumps({"type": "error", "error": f"The model `{msg.get('model')}` does not exist.",
                                              "code": "model_not_found"}))
                else:
                    validated = True
            elif t == "input_audio_buffer.append":
                appends += 1
                n = len(base64.b64decode(msg["audio"])) // 2
                before = samples // 8000
                samples += n
                self.total_samples += n
                for k in range(before, samples // 8000):
                    piece = f"句子{k}。"
                    text += piece
                    await ws.send(json.dumps({"type": "transcription.delta", "delta": piece}))
                if self.drop_after and conn_no == 1 and appends >= self.drop_after:
                    await ws.close()
                    return
            elif t == "input_audio_buffer.commit":
                if not validated:
                    await ws.send(json.dumps({"type": "error", "error": "Model not validated", "code": "model_not_validated"}))
                elif msg.get("final"):
                    await ws.send(json.dumps({"type": "transcription.done", "text": text}))

    def process_request(self, connection, request):
        if self.plain_http:
            return connection.respond(404, "not here\n")
        if self.key and request.headers.get("Authorization") != f"Bearer {self.key}":
            return connection.respond(401, "unauthorized\n")
        return None

    def start(self):
        def run():
            async def main():
                async with websockets.serve(self.handler, "127.0.0.1", 0, process_request=self.process_request) as srv:
                    self.port = srv.sockets[0].getsockname()[1]
                    self._ready.set()
                    await asyncio.Future()
            asyncio.run(main())

        threading.Thread(target=run, daemon=True).start()
        self._ready.wait(5)
        return f"ws://127.0.0.1:{self.port}/v1/realtime"


def run(coro):
    return asyncio.run(coro)


def test_sentence_splitter():
    s = SentenceSplitter()
    assert s.feed("你好") == ([], "你好")
    assert s.feed("你好。今天") == (["你好。"], "今天")
    assert s.feed("你好。今天3.5度！好") == (["今天3.5度！"], "好")
    assert s.flush("你好。今天3.5度！好的") == "好的"


def test_validate_success():
    url = FakeVllmServer(key="k123").start()
    r = run(validate_async("vllm_realtime", url, "k123", "Qwen/Qwen3-ASR-0.6B"))
    assert r.ok and "句子0。" in r.text and r.first_text_latency is not None and r.total < 10


def test_validate_failures():
    srv = FakeVllmServer(key="k123")
    url = srv.start()
    r = run(validate_async("vllm_realtime", url, "wrong", "Qwen/Qwen3-ASR-0.6B"))
    assert not r.ok and r.kind == "auth"
    r = run(validate_async("vllm_realtime", url, "k123", "other-model"))
    assert not r.ok and r.kind == "model"
    r = run(validate_async("vllm_realtime", "ws://127.0.0.1:1/v1/realtime", "", "m"))
    assert not r.ok and r.kind == "connect"
    url2 = FakeVllmServer(plain_http=True).start()
    r = run(validate_async("vllm_realtime", url2, "", "m"))
    assert not r.ok and r.kind == "protocol"


def feed_audio(sess, seconds, block=0.1, pace=0.0):
    n = int(16000 * block)
    for _ in range(int(seconds / block)):
        sess.feed(np.full(n, 0.01, np.float32))
        if pace:
            time.sleep(pace)


def collect():
    finals, partials, errors, stopped = [], [], [], []
    cb = SessionCallbacks(on_final=finals.append, on_partial=partials.append, on_error=errors.append,
                          on_stopped=stopped.append)
    return cb, finals, partials, errors, stopped


def test_session_records_and_writes(tmp_path):
    url = FakeVllmServer().start()
    cb, finals, partials, errors, stopped = collect()
    s = RealtimeSession(SessionSettings("vllm_realtime", url, "", "Qwen/Qwen3-ASR-0.6B", None, chunk_ms=100),
                        tmp_path, "live", cb)
    s.start()
    feed_audio(s, 3.0, pace=0.005)
    s.stop()
    assert not errors
    assert [f["text"] for f in finals][:2] == ["句子0。", "句子1。"] and len(finals) == 6
    assert all(b["start"] >= a["end"] - 1e-6 for a, b in zip(finals, finals[1:]))
    info = stopped[0]
    import soundfile as sf

    assert sf.info(info["wav"]).duration == pytest.approx(3.0, abs=0.01)
    assert "句子5。" in (tmp_path / "live.txt").read_text() and "-->" in (tmp_path / "live.srt").read_text()


def test_session_reconnects_and_replays(tmp_path):
    srv = FakeVllmServer(drop_after=10)
    url = srv.start()
    cb, finals, partials, errors, stopped = collect()
    s = RealtimeSession(SessionSettings("vllm_realtime", url, "", "Qwen/Qwen3-ASR-0.6B", None, chunk_ms=100,
                                        reconnect_delay=0.05, max_reconnects=3), tmp_path, "rc", cb)
    s.start()
    feed_audio(s, 4.0, pace=0.03)
    s.stop()
    assert srv.connections >= 2 and s.state.reconnects >= 1
    # every captured sample reached the server (buffered audio replayed; <= 0.5 s may be sent twice)
    assert 4.0 * 16000 <= srv.total_samples <= 4.5 * 16000 + 1600
    import soundfile as sf

    assert sf.info(stopped[0]["wav"]).duration == pytest.approx(4.0, abs=0.01)


def test_session_server_error_keeps_audio(tmp_path):
    url = FakeVllmServer().start()
    cb, finals, partials, errors, stopped = collect()
    s = RealtimeSession(SessionSettings("vllm_realtime", url, "", "bad-model", None), tmp_path, "err", cb)
    s.start()
    feed_audio(s, 1.0)
    s.stop()
    assert errors and stopped[0]["error"]
    assert (tmp_path / "err.wav").exists()


def test_realtime_disabled_until_validated():
    c = AppConfig()
    assert not c.realtime_active
    c.realtime.url = "ws://x/v1/realtime"
    assert not c.realtime_active
    c.realtime.validated = True
    assert c.realtime_active
    c.realtime.enabled = False
    assert not c.realtime_active


def test_first_sentence_after_reconnect_is_deduplicated(tmp_path):
    finals = []
    s = RealtimeSession(SessionSettings("vllm_realtime", "ws://unused", "", "m", None), tmp_path, "d",
                        SessionCallbacks(on_final=finals.append))
    s._on_final("我们明天去公园散步。")
    s._dedupe_next = True  # as set by a successful reconnect
    s._on_final("公园散步。然后吃饭。")
    s._on_final("公园很大。")  # only the first sentence after reconnect is checked
    assert [f["text"] for f in finals] == ["我们明天去公园散步。", "然后吃饭。", "公园很大。"]
