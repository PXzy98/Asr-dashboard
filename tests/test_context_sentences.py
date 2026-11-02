from asr_dashboard.config.schema import Glossary
from asr_dashboard.core.context import build_context
from asr_dashboard.core.sentences import attach_punctuation, split_sentences
from asr_dashboard.models import Word


def test_context_merge_order_and_dedup():
    g1 = Glossary(name="产品", description="公司产品名", terms=["通义千问", "Qwen3-ASR", "百炼"])
    g2 = Glossary(name="人名", description="", terms=["张三", "qwen3-asr", "李四"])
    r = build_context("这是一次技术评审会议", [g1, g2], "参会人：王五\n这是一次技术评审会议", 2000)
    lines = r.text.splitlines()
    assert lines[0] == "这是一次技术评审会议"
    assert lines[1] == "公司产品名"
    assert lines[2] == "参会人：王五"
    assert lines[-1] == "术语：通义千问、Qwen3-ASR、百炼、张三、李四"
    assert not r.truncated and r.warnings == []


def test_context_truncation_warns():
    g = Glossary(terms=[f"术语{i:04d}" for i in range(500)])
    r = build_context("", [g], "", 300)
    assert r.truncated and len(r.text) <= 300
    assert r.warnings and "截断" in r.warnings[0]
    assert not r.text.endswith("、")


def test_empty_context():
    assert build_context().text == ""


def test_attach_punctuation_and_split():
    text = "今天天气很好。我们去公园吧！好的"
    items = [(c, i * 0.3, i * 0.3 + 0.25) for i, c in enumerate("今天天气很好我们去公园吧好的")]
    words = attach_punctuation(text, items)
    assert words[5].text == "好。" and words[11].text == "吧！"
    segs = split_sentences(words, max_duration=15)
    assert [s.text for s in segs] == ["今天天气很好。", "我们去公园吧！", "好的"]
    assert segs[0].start == 0 and segs[1].start > segs[0].end - 1e-9


def test_split_on_pause_and_max_duration():
    words = [Word("hello", 0, 0.4), Word("there", 0.5, 0.9), Word("after", 2.0, 2.3), Word("pause", 2.4, 2.8)]
    segs = split_sentences(words, max_duration=15, pause=0.6)
    assert [s.text for s in segs] == ["hello there", "after pause"]
    many = [Word(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(60)]
    segs = split_sentences(many, max_duration=10)
    assert all(s.end - s.start <= 10 for s in segs) and len(segs) >= 3


def test_english_punct():
    text = "Hello, world. How are you?"
    items = [("Hello", 0, 0.3), ("world", 0.4, 0.7), ("How", 1.0, 1.1), ("are", 1.2, 1.3), ("you", 1.4, 1.6)]
    words = attach_punctuation(text, items)
    assert [w.text for w in words] == ["Hello,", "world.", "How", "are", "you?"]
    segs = split_sentences(words)
    assert [s.text for s in segs] == ["Hello, world.", "How are you?"]
