import json
import re

from asr_dashboard.export import writers as W
from asr_dashboard.export.streaming import StreamingWriter
from asr_dashboard.models import Segment, SpeakerInfo, TranscriptionResult, Word
from asr_dashboard.text import rules as R


def result():
    segs = [Segment("你好，欢迎。", 0.0, 1.5, "SPEAKER_00", "zh", [Word("你好，", 0, 0.6)]),
            Segment("Thanks for having me.", 1.6, 3.2554, "SPEAKER_01", "en"),
            Segment("", 3.3, 3.4)]
    return TranscriptionResult("demo.mp4", segs, "", "zh", 3.4,
                               speakers={"SPEAKER_00": SpeakerInfo("SPEAKER_00", "张三", "p1", 0.81),
                                         "SPEAKER_01": SpeakerInfo("SPEAKER_01")},
                               overlaps=[(1.0, 1.2, ["SPEAKER_00", "SPEAKER_01"])], summary="- 要点一")


def test_time_formats():
    assert W.fmt_srt(3725.0456) == "01:02:05,046"
    assert W.fmt_vtt(0.5) == "00:00:00.500"
    assert W.fmt_ass(62.346) == "0:01:02.35"


def test_srt_vtt_txt():
    r = result()
    o = W.ExportOptions()
    srt = W.to_srt(r, o)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\n[张三] 你好，欢迎。\n\n2\n")
    assert "[SPEAKER_01] Thanks" in srt and srt.count("-->") == 2
    vtt = W.to_vtt(r, o)
    assert vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n")
    assert re.search(r"00:00:01\.600 --> 00:00:03\.255", vtt)
    txt = W.to_txt(r, o)
    assert txt.splitlines()[0] == "[00:00:00] 张三: 你好，欢迎。"
    assert W.to_txt(r, W.ExportOptions(include_speakers=False, txt_timestamps=False)).splitlines()[1] == "Thanks for having me."


def test_ass_header_and_colors():
    ass = W.to_ass(result(), W.ExportOptions(ass_font="Noto Sans", ass_font_size=36))
    assert ass.startswith("[Script Info]\nScriptType: v4.00+")
    assert "[V4+ Styles]" in ass and "[Events]" in ass
    assert "Style: S0,Noto Sans,36," in ass and "Style: S1,Noto Sans,36," in ass
    assert "Dialogue: 0,0:00:00.00,0:00:01.50,S0,张三," in ass


def test_json_complete_and_md():
    d = json.loads(W.to_json(result()))
    for k in ("segments", "speakers", "overlaps", "engine_settings", "context_used", "postprocess_mode",
              "summary", "timings", "warnings"):
        assert k in d
    assert d["segments"][0]["words"][0]["text"] == "你好，"
    assert TranscriptionResult.from_dict(d).speakers["SPEAKER_00"].display_name == "张三"
    assert "要点一" in W.to_markdown(result())


def test_save_outputs_rename_and_skip(tmp_path):
    r = result()
    a = W.save_outputs(r, tmp_path, "demo", ["srt", "txt", "md"], W.ExportOptions())
    assert set(a) == {"srt", "txt", "md"}
    b = W.save_outputs(r, tmp_path, "demo", ["srt"], W.ExportOptions(), on_existing="rename")
    assert b["srt"].name == "demo (1).srt"
    c = W.save_outputs(r, tmp_path, "demo", ["srt"], W.ExportOptions(), on_existing="skip")
    assert c == {}


def test_streaming_writer_survives_partial(tmp_path):
    w = StreamingWriter(tmp_path, "live", ["txt", "srt", "json"], W.ExportOptions())
    w.write_segments([Segment("first.", 0, 1, "SPEAKER_00")], {"SPEAKER_00": "Ann"})
    # simulate crash: files already contain the first chunk
    assert "Ann: first." in (tmp_path / "live.txt").read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:01,000\n[Ann] first." in (tmp_path / "live.srt").read_text(encoding="utf-8")
    assert (tmp_path / "live.jsonl.part").exists()
    w.write_segments([Segment("second.", 1, 2)])
    paths = w.finalize()
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert [s["text"] for s in data["segments"]] == ["first.", "second."]
    assert not (tmp_path / "live.jsonl.part").exists()


def test_rules_punctuation_whitespace_caps():
    o = R.RuleOptions()
    assert R.clean_text("今天 天气 很好,我们去 公园吧.", "zh", o) == "今天天气很好，我们去公园吧。"
    assert R.clean_text("我用iPhone15拍的，，好看吗??", "zh", o) == "我用 iPhone15 拍的，好看吗？"
    assert R.clean_text("hello world ,this is great。 really？", "en", o) == "Hello world, this is great. Really?"
    assert R.clean_text("价格是3.5元", "zh", o) == "价格是 3.5 元"


def test_fillers():
    o = R.RuleOptions(remove_fillers=True, filler_words={"zh": ["嗯", "啊", "呃", "那个", "就是"],
                                                          "en": ["um", "uh", "you know"], "ja": ["えーと", "あの"]})
    assert R.clean_text("嗯，那个，我们明天开会吧", "zh", o) == "我们明天开会吧"
    assert R.clean_text("这个就是答案", "zh", o) == "这个就是答案"
    assert R.clean_text("um, I think, you know, it works", "en", o) == "I think, it works"
    assert R.clean_text("えーと、あの、はい", "ja", o) == "はい"


def test_merge_respects_speakers_and_keeps_original():
    segs = [Segment("我们今天", 0, 1, "A", "zh"), Segment("讨论预算。", 1.2, 2, "A", "zh"),
            Segment("好的", 2.1, 3, "B", "zh"), Segment("没问题", 3.1, 4, "A", "zh")]
    out = R.apply_rules(segs, R.RuleOptions())
    assert [s.text for s in out] == ["我们今天讨论预算。", "好的", "没问题"]
    assert out[0].start == 0 and out[0].end == 2
