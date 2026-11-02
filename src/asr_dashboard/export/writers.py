"""TXT / SRT / VTT / JSON / ASS / Markdown writers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from asr_dashboard.models import Segment, TranscriptionResult

FORMATS = ("txt", "srt", "vtt", "json", "ass", "md")
SPEAKER_COLORS = ["FFFFFF", "00FFFF", "FFD700", "7CFC00", "FF69B4", "87CEFA", "FFA500", "DA70D6", "ADFF2F",
                  "F08080"]


@dataclass
class ExportOptions:
    include_speakers: bool = True
    txt_timestamps: bool = True
    ass_font: str = "Arial"
    ass_font_size: int = 20

    @classmethod
    def from_config(cls, out) -> ExportOptions:
        return cls(out.include_speakers, out.txt_timestamps, out.ass_font, out.ass_font_size)


def fmt_srt(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_vtt(t: float) -> str:
    return fmt_srt(t).replace(",", ".")


def fmt_ass(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def fmt_clock(t: float) -> str:
    s = int(t)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _name(result: TranscriptionResult | None, seg: Segment) -> str | None:
    if not seg.speaker:
        return None
    return result.speaker_name(seg.speaker) if result else seg.speaker


def _subtitle_segments(segments: list[Segment]) -> list[Segment]:
    """Guarantee strictly usable cues: drop empty text, ensure end > start and monotonic order."""
    out = []
    for s in sorted((s for s in segments if s.text.strip()), key=lambda s: (s.start, s.end)):
        end = s.end if s.end > s.start else s.start + 0.5
        out.append(Segment(s.text, s.start, end, s.speaker, s.language, s.words, s.original_text))
    return out


def to_txt(result: TranscriptionResult, opts: ExportOptions) -> str:
    lines = []
    for seg in result.segments:
        if not seg.text.strip():
            continue
        prefix = ""
        if opts.txt_timestamps:
            prefix += f"[{fmt_clock(seg.start)}] "
        name = _name(result, seg) if opts.include_speakers else None
        if name:
            prefix += f"{name}: "
        lines.append(prefix + seg.text.strip())
    return "\n".join(lines) + ("\n" if lines else "")


def srt_cue(i: int, seg: Segment, name: str | None) -> str:
    text = f"[{name}] {seg.text.strip()}" if name else seg.text.strip()
    return f"{i}\n{fmt_srt(seg.start)} --> {fmt_srt(seg.end)}\n{text}\n\n"


def to_srt(result: TranscriptionResult, opts: ExportOptions) -> str:
    return "".join(srt_cue(i, s, _name(result, s) if opts.include_speakers else None)
                   for i, s in enumerate(_subtitle_segments(result.segments), 1))


def to_vtt(result: TranscriptionResult, opts: ExportOptions) -> str:
    parts = ["WEBVTT\n\n"]
    for s in _subtitle_segments(result.segments):
        name = _name(result, s) if opts.include_speakers else None
        text = f"[{name}] {s.text.strip()}" if name else s.text.strip()
        parts.append(f"{fmt_vtt(s.start)} --> {fmt_vtt(s.end)}\n{text}\n\n")
    return "".join(parts)


def to_json(result: TranscriptionResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)


def to_markdown(result: TranscriptionResult) -> str | None:
    if not result.summary:
        return None
    title = Path(result.source_file).name or "转录"
    return f"# {title} — 摘要\n\n{result.summary.strip()}\n"


def _ass_color(hex_rgb: str) -> str:
    r, g, b = hex_rgb[0:2], hex_rgb[2:4], hex_rgb[4:6]
    return f"&H00{b}{g}{r}"


def to_ass(result: TranscriptionResult, opts: ExportOptions) -> str:
    speakers = sorted({s.speaker for s in result.segments if s.speaker})
    style_of = {spk: f"S{i}" for i, spk in enumerate(speakers)}
    font, size = opts.ass_font, opts.ass_font_size
    styles = [f"Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,1,2,20,20,20,1"]
    for st in style_of.values():
        color = _ass_color(SPEAKER_COLORS[int(st[1:]) % len(SPEAKER_COLORS)])
        styles.append(f"Style: {st},{font},{size},{color},&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,1,2,20,20,20,1")
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding\n" + "\n".join(styles) + "\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = []
    for s in _subtitle_segments(result.segments):
        name = _name(result, s) if opts.include_speakers else ""
        style = style_of.get(s.speaker, "Default") if s.speaker else "Default"
        text = s.text.strip().replace("\n", "\\N")
        if name and opts.include_speakers:
            text = f"[{name}] {text}"
        lines.append(f"Dialogue: 0,{fmt_ass(s.start)},{fmt_ass(s.end)},{style},{name or ''},0,0,0,,{text}")
    return header + "\n".join(lines) + "\n"


def render(result: TranscriptionResult, fmt: str, opts: ExportOptions) -> str | None:
    if fmt == "txt":
        return to_txt(result, opts)
    if fmt == "srt":
        return to_srt(result, opts)
    if fmt == "vtt":
        return to_vtt(result, opts)
    if fmt == "json":
        return to_json(result)
    if fmt == "ass":
        return to_ass(result, opts)
    if fmt == "md":
        return to_markdown(result)
    raise ValueError(f"未知导出格式 {fmt}")


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    i = 1
    while True:
        cand = path.with_name(f"{stem} ({i}){suffix}")
        if not cand.exists():
            return cand
        i += 1


def output_path(out_dir: Path, stem: str, fmt: str, on_existing: str = "rename") -> Path | None:
    p = Path(out_dir) / f"{stem}.{fmt}"
    if p.exists():
        if on_existing == "skip":
            return None
        if on_existing == "rename":
            return unique_path(p)
    return p


def save_outputs(result: TranscriptionResult, out_dir: str | Path, stem: str, formats: list[str], opts: ExportOptions,
                 on_existing: str = "rename") -> dict[str, Path]:
    """Write all formats; returns {fmt: path}. ``on_existing``: rename (default) / overwrite / skip."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for fmt in formats:
        content = render(result, fmt, opts)
        if content is None:
            continue
        p = output_path(out_dir, stem, fmt, on_existing)
        if p is None:
            continue
        tmp = p.with_name(p.name + ".part")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(p)
        written[fmt] = p
    return written
