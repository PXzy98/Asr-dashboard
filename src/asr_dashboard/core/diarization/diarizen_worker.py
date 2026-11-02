"""Runs INSIDE the isolated DiariZen environment (Python 3.10, torch 2.1, DiariZen's pyannote fork).

Protocol (one JSON object per line):
  stdin  : {"wav": path, "num_speakers": int, "min_speakers": int, "max_speakers": int,
            "device": "cpu"|"cuda:N", "model_id": str}
  stdout : {"type": "progress", "value": 0..1, "message": str}
           {"type": "result", "turns": [{"start": s, "end": e, "speaker": str}], "embeddings": {label: [...]}}
           {"type": "error", "message": str}

This file must not import anything from asr_dashboard (the package is not installed in that env).
"""

import json
import sys
import traceback


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    try:
        req = json.loads(sys.stdin.readline() or "{}")
        emit({"type": "progress", "value": 0.05, "message": "加载 DiariZen 模型"})
        import torch
        from diarizen.pipelines.inference import DiariZenPipeline

        model_id = req.get("model_id") or "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
        pipe = DiariZenPipeline.from_pretrained(model_id)
        device = req.get("device") or "cpu"
        try:
            pipe.to(torch.device(device))
        except Exception:  # some versions keep the device internally
            pass
        emit({"type": "progress", "value": 0.2, "message": "运行 DiariZen"})
        kwargs = {}
        n = int(req.get("num_speakers") or 0)
        if n > 0:
            kwargs["num_speakers"] = n
        else:
            if req.get("min_speakers"):
                kwargs["min_speakers"] = int(req["min_speakers"])
            if req.get("max_speakers"):
                kwargs["max_speakers"] = int(req["max_speakers"])
        try:
            result = pipe(req["wav"], **kwargs)
        except TypeError:  # older pipelines don't accept speaker-count hints
            result = pipe(req["wav"])
        ann = getattr(result, "speaker_diarization", result)
        turns = [{"start": float(t.start), "end": float(t.end), "speaker": str(spk)}
                 for t, _, spk in ann.itertracks(yield_label=True)]
        emit({"type": "result", "turns": turns, "embeddings": {}})
    except Exception as e:  # noqa: BLE001
        emit({"type": "error", "message": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
        sys.exit(1)


if __name__ == "__main__":
    main()
