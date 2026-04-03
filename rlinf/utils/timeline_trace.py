# rlinf/utils/timeline_trace.py
import json
import os
import threading
import time

_lock = threading.Lock()

def append_timeline_event(cfg, *, component: str, rank: int, tag: str,
                         t0: float, t1: float, global_step: int | None = None,
                         extra: dict | None = None):
    if not cfg.runner.get("timeline_trace", False):
        return
    base = cfg.runner.get("timeline_dir") or os.path.join(cfg.runner.logger.log_path, "timeline")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"{component}_rank{rank}.jsonl")
    rec = {
        "t0": t0,
        "t1": t1,
        "component": component,
        "rank": rank,
        "tag": tag,
        "global_step": global_step,
    }
    if extra:
        rec.update(extra)
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)