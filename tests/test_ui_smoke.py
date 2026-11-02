"""Offscreen smoke tests: the window builds, pages work, settings round-trip through config.yaml."""

import numpy as np
import pytest

pytest.importorskip("qfluentwidgets")

from asr_dashboard.config.manager import ConfigManager  # noqa: E402
from asr_dashboard.config.schema import ApiModelConfig  # noqa: E402
from asr_dashboard.config.secrets import SecretStore, api_key_name  # noqa: E402
from asr_dashboard.models import Segment, SpeakerInfo, TranscriptionResult  # noqa: E402


@pytest.fixture
def services(tmp_path):
    from asr_dashboard.services import Services

    cm = ConfigManager(tmp_path / "config.yaml", SecretStore(tmp_path / "secrets.yaml", use_keyring=False))
    return Services(cm)


@pytest.fixture
def window(qtbot, services):
    from asr_dashboard.ui.main_window import MainWindow

    w = MainWindow(services)
    qtbot.addWidget(w)
    w.show()
    return w


def test_engine_dropdown_only_qwen_local_plus_api(window, services):
    combo = window.home.options.engine
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert labels == ["本地 · Qwen3-ASR 1.7B", "本地 · Qwen3-ASR 0.6B"]
    services.config.update(lambda c: c.api_models.append(ApiModelConfig(id="g1", name="Groq")))
    window.refresh_all()
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert labels[-1] == "API · Groq" and len(labels) == 3


def test_realtime_nav_disabled_until_validated(window, services):
    item = window.navigationInterface.widget(window.realtime.objectName())
    assert not item.isEnabled()
    services.config.update(lambda c: (setattr(c.realtime, "url", "ws://x/v1/realtime"), setattr(c.realtime, "validated", True)))
    window.update_realtime_nav()
    assert item.isEnabled()


def test_qwen_settings_saved_and_reloaded(window, services, tmp_path):
    form = window.settings.forms["qwen3"]
    form.widgets["max_new_tokens"].setValue(2048)
    form.widgets["max_inference_batch_size"].setValue(8)
    idx = [form.widgets["variant"].itemData(i) for i in range(form.widgets["variant"].count())].index("0.6b")
    form.widgets["variant"].setCurrentIndex(idx)
    form.widgets["aligner.enabled"].setChecked(False)
    form.widgets["context"].setPlainText("术语：千问")
    window.settings.save_section("qwen3")
    again = ConfigManager(tmp_path / "config.yaml", services.secrets).config
    assert again.qwen3.max_new_tokens == 2048 and again.qwen3.max_inference_batch_size == 8
    assert again.qwen3.variant == "0.6b" and again.qwen3.aligner.enabled is False and again.qwen3.context == "术语：千问"


def test_engine_settings_dialog_saves_api_key_outside_yaml(qtbot, services, tmp_path):
    from asr_dashboard.ui.widgets.task_options import EngineSettingsDialog

    services.config.update(lambda c: c.api_models.append(ApiModelConfig(id="o1", name="OpenAI")))
    dlg = EngineSettingsDialog(services, "o1")
    qtbot.addWidget(dlg)
    dlg.form.widgets["api_key"].setText("sk-abcdefghijklmnop1234")
    dlg.form.widgets["model"].setText("whisper-1")
    dlg.save()
    assert services.cfg.api_model("o1").model == "whisper-1"
    assert services.secrets.get(api_key_name("o1")) == "sk-abcdefghijklmnop1234"
    assert "sk-abcdefghijklmnop1234" not in (tmp_path / "config.yaml").read_text(encoding="utf-8")


def test_results_page_rename_and_undo(window):
    r = TranscriptionResult("x.wav", [Segment("你好。", 0, 1, "SPEAKER_00", original_text="你好"),
                                      Segment("hi", 1, 2, "SPEAKER_01")], "", "zh", 2,
                            speakers={"SPEAKER_00": SpeakerInfo("SPEAKER_00", None, None, 0.42),
                                      "SPEAKER_01": SpeakerInfo("SPEAKER_01", "李四", "p", 0.9)},
                            warnings=["说话人分离失败，已跳过：x"], summary="## 要点")
    page = window.results
    page.show_result(r)
    assert page.table.rowCount() == 2 and page.table.item(1, 2).text() == "李四"
    assert page.warn_list.count() == 1 and not page.summary_card.isHidden()
    r.speakers["SPEAKER_00"].display_name = "张三"
    page.fill_table()
    assert page.table.item(0, 2).text() == "张三"
    assert "<del>" in page._diff_html() or "<ins>" in page._diff_html()
    page.undo_all()
    assert r.segments[0].text == "你好"


def test_schema_form_validation():
    from asr_dashboard.engines.qwen3.engine import Qwen3LocalEngine
    from asr_dashboard.ui.widgets.schema_form import SchemaForm

    f = SchemaForm(Qwen3LocalEngine.settings_schema(), {"source": "huggingface", "backend": "qwen_asr"})
    assert f.form.isRowVisible(f.widgets["local_path__row"]) is False
    f.widgets["source"].setCurrentIndex(2)  # local
    assert f.form.isRowVisible(f.widgets["local_path__row"]) is True
    vals = f.values()
    assert vals["aligner"]["enabled"] is True and vals["max_new_tokens"] == 1024


def test_batch_page_scan(window, tmp_path):
    d = tmp_path / "in"
    (d / "a").mkdir(parents=True)
    for n in ("10.mp3", "2.mp3", "a/1.wav"):
        (d / n).write_bytes(b"x")
    window.batch.folder = str(d)
    window.batch.scan()
    items = [window.batch.preview.item(i).text() for i in range(window.batch.preview.count())]
    assert [i.replace("\\", "/") for i in items] == ["2.mp3", "10.mp3", "a/1.wav"]


def test_voiceprints_page_lists(window, services):
    p = services.voiceprints.add_person("Alice")
    services.voiceprints.add_sample(p.id, np.zeros(16000, np.float32))
    window.voiceprints.refresh()
    assert window.voiceprints.people.item(0, 0).text() == "Alice"


def _fake_openai_server():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    hits = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("content-length", 0))
            self.rfile.read(n)
            hits.append((self.path, self.headers.get("authorization")))
            body = json.dumps({"text": "hello world.", "language": "english",
                               "segments": [{"start": 0.2, "end": 1.4, "text": "hello world."}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}/v1", hits, srv


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg")
def test_api_model_test_connection_and_home_transcription(qtbot, window, services, tmp_path):
    import soundfile as sf

    url, hits, srv = _fake_openai_server()
    page = window.api_models
    page.preset.setCurrentIndex(0)
    page.create()
    page.form.widgets["base_url"].setText(url)
    page.form.widgets["model"].setText("whisper-1")
    page.form.widgets["api_key"].setText("sk-local-test-123456789")
    page.save()
    m = services.cfg.api_models[0]
    assert m.base_url == url
    page.test()
    qtbot.waitUntil(lambda: "耗时" in page.result.toPlainText() or "失败" in page.result.toPlainText(), timeout=15000)
    assert "hello world." in page.result.toPlainText() and hits[-1][1] == "Bearer sk-local-test-123456789"

    services.config.update(lambda c: setattr(c.app, "output_dir", str(tmp_path / "out")))
    wav = tmp_path / "clip.wav"
    sf.write(wav, (0.1 * np.random.default_rng(0).standard_normal(16000 * 3)).astype("float32"), 16000)
    home = window.home
    home.refresh()
    idx = next(i for i in range(home.options.engine.count()) if home.options.engine.itemData(i) == m.id)
    home.options.engine.setCurrentIndex(idx)
    home.file = str(wav)
    home.start()
    qtbot.waitUntil(lambda: not home.is_busy() and home.start_btn.isEnabled(), timeout=30000)
    assert window.results.table.rowCount() == 1 and window.results.table.item(0, 3).text().startswith("Hello world")
    assert (tmp_path / "out" / "clip.srt").exists()
    cfg_text = services.config.path.read_text(encoding="utf-8")
    assert "sk-local-test" not in cfg_text
    srv.shutdown()


def test_batch_worker_with_api_engine_finishes(qtbot, services, tmp_path):
    # review finding: API configs have no keep_loaded -> AttributeError after a successful batch
    import soundfile as sf

    from asr_dashboard.core import batch as B
    from asr_dashboard.core.tasks import BatchWorker

    url, hits, srv = _fake_openai_server()
    services.config.update(lambda c: c.api_models.append(ApiModelConfig(id="b1", name="B", base_url=url, model="m")))
    d = tmp_path / "in"
    d.mkdir()
    sf.write(d / "1.wav", np.zeros(16000 * 2, np.float32) + 0.01, 16000)
    m = B.BatchManifest.create(d, B.scan_folder(d), tmp_path / "out", {"formats": ["srt"]})
    cfg = services.cfg.model_copy(deep=True)
    cfg.output.formats = ["srt"]
    w = BatchWorker(services, "b1", cfg, m)
    failed, finished = [], []
    w.failed.connect(lambda msg, tb: failed.append(tb))
    w.finished_batch.connect(finished.append)
    with qtbot.waitSignal(w.finished, timeout=30000):
        w.start()
    qtbot.wait(50)
    assert not failed, failed and failed[0]
    assert finished and m.items[0].status == B.DONE
    srv.shutdown()
