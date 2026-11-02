import json
import os
import shutil
import time
from pathlib import Path

import pytest

from asr_dashboard.core import batch as B
from asr_dashboard.models import CancelledError


def make_tree(root: Path):
    (root / "sub").mkdir(parents=True)
    (root / ".hidden").mkdir()
    for i in range(1, 21):
        d = root / ("sub" if i > 15 else "")
        (d / f"{i}.mp3").write_bytes(b"x" * (100 + i))
    (root / "broken.wav").write_bytes(b"garbage")
    (root / "empty.mp3").write_bytes(b"")
    (root / ".hidden" / "h.mp3").write_bytes(b"x")
    (root / "notes.txt").write_text("no", encoding="utf-8")
    return root


def test_natural_sort_and_filters(tmp_path):
    root = make_tree(tmp_path / "in")
    files = B.scan_folder(root)
    rels = [f.relative_to(root).as_posix() for f in files]
    assert rels[:3] == ["1.mp3", "2.mp3", "3.mp3"]
    assert rels.index("10.mp3") == 9 and "15.mp3" in rels and rels[-1] == "sub/20.mp3"
    assert "empty.mp3" not in rels and ".hidden/h.mp3" not in rels and "notes.txt" not in rels
    assert "broken.wav" in rels
    assert len(B.scan_folder(root, recursive=False)) == 16
    assert B.natural_key("file2") < B.natural_key("file10")


def fake_process(fail_names=(), calls=None, crash_after=None):
    def process(item, out_dir: Path, on_existing, prog, cancel):
        if calls is not None:
            calls.append(item.rel)
        if crash_after is not None and len(calls) > crash_after:
            raise KeyboardInterrupt  # simulated hard kill of the process
        if Path(item.path).name in fail_names:
            raise RuntimeError("FFmpeg 处理失败（文件可能已损坏）")
        prog(0.5, "half")
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{Path(item.rel).stem}.srt"
        p.write_text("1\n00:00:00,000 --> 00:00:01,000\nx\n", encoding="utf-8")
        t = out_dir / f"{Path(item.rel).stem}.txt"
        t.write_text("x", encoding="utf-8")
        return B.ProcessOutcome({"srt": str(p), "txt": str(t)}, 1.0, ["张三"], 0)
    return process


def manifest(tmp_path):
    root = make_tree(tmp_path / "in")
    files = B.scan_folder(root)
    return B.BatchManifest.create(root, files, tmp_path / "out", {"formats": ["srt", "txt"]})


def test_sequential_run_with_failure_and_reports(tmp_path):
    m = manifest(tmp_path)
    calls = []
    B.BatchRunner(m, fake_process({"broken.wav"}, calls), prevent_sleep=False).run()
    assert calls == [i.rel for i in m.items]  # strict order
    st = {i.rel: i.status for i in m.items}
    assert st["broken.wav"] == B.FAILED and sum(v == B.DONE for v in st.values()) == 20
    assert (tmp_path / "out" / "sub" / "20.srt").exists()  # mirrored structure
    rep = json.loads((tmp_path / "out" / "batch_report.json").read_text(encoding="utf-8"))
    assert rep["done"] == 20 and rep["failed"] == 1
    assert "broken.wav" in (tmp_path / "out" / "batch_report.csv").read_text(encoding="utf-8-sig")
    assert m.finished_at is not None and not m.unfinished


def test_crash_and_resume(tmp_path):
    m = manifest(tmp_path)
    calls = []
    with pytest.raises(KeyboardInterrupt):
        B.BatchRunner(m, fake_process(calls=calls, crash_after=7), prevent_sleep=False).run()
    # stray temp file from an interrupted write must not break loading
    Path(m.path + ".999.tmp").write_text("{ half", encoding="utf-8")
    reloaded = B.BatchManifest.load(m.path)
    assert reloaded.unfinished and reloaded.next_index() == 7
    assert [i.status for i in reloaded.items[:7]] == [B.DONE] * 7
    assert any(x.batch_id == m.batch_id for x in B.find_unfinished())
    calls2 = []
    B.BatchRunner(reloaded, fake_process({"broken.wav"}, calls2), prevent_sleep=False).run()
    assert calls2[0] == reloaded.items[7].rel and len(calls2) == 21 - 7
    assert not B.BatchManifest.load(m.path).unfinished


def test_completed_detection(tmp_path):
    m = manifest(tmp_path)
    B.BatchRunner(m, fake_process({"broken.wav"}), prevent_sleep=False).run()
    item = m.items[0]
    assert B.is_completed(item)
    Path(item.outputs["srt"]).unlink()
    assert not B.is_completed(item)
    item2 = m.items[1]
    time.sleep(0.01)
    os.utime(item2.path, (time.time() + 5, time.time() + 5))
    assert not B.is_completed(item2)


def test_existing_outputs_skip_and_retry_failed(tmp_path):
    m = manifest(tmp_path)
    B.BatchRunner(m, fake_process({"broken.wav"}), prevent_sleep=False).run()
    m2 = B.BatchManifest.create(m.folder, [Path(i.path) for i in m.items], m.output_dir, {"formats": ["srt", "txt"]})
    calls = []
    r = B.BatchRunner(m2, fake_process({"broken.wav"}, calls), on_existing="skip", prevent_sleep=False)
    r.run()
    assert calls == ["broken.wav"] and sum(i.status == B.SKIPPED for i in m2.items) == 20
    r.retry_failed()
    calls.clear()
    r.process = fake_process((), calls)
    r.run()
    assert calls == ["broken.wav"] and m2.items[-1 if m2.items[-1].rel == "broken.wav" else 15].status in (B.DONE, B.SKIPPED)


def test_pause_resume_cancel(tmp_path):
    m = manifest(tmp_path)
    calls = []
    runner = None

    def process(item, out_dir, on_ex, prog, cancel):
        calls.append(item.rel)
        if len(calls) == 2:
            runner.pause()
            import threading

            threading.Timer(0.2, runner.resume).start()
        if len(calls) == 4:
            runner.stop()
            raise CancelledError()
        return fake_process()(item, out_dir, on_ex, prog, cancel)

    runner = B.BatchRunner(m, process, prevent_sleep=False)
    runner.run()
    assert len(calls) == 4 and m.items[3].status == B.CANCELLED and m.items[4].status == B.PENDING
    assert m.unfinished


def test_abandon(tmp_path):
    m = manifest(tmp_path)
    m.save()
    B.abandon(m)
    assert not B.BatchManifest.load(m.path).unfinished
    shutil.rmtree(tmp_path / "out")
