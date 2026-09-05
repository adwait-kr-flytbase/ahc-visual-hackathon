"""Per-video timing -> the runtime_metadata block the submission asks for."""
import time
from collections import defaultdict


class Timer:
    def __init__(self):
        self.calls = defaultdict(list)
        self.t0 = time.perf_counter()
        self.frames = 0
        self.chunks = 0

    def track(self, name):
        return _Span(self, name)

    def record(self, name, ms):
        self.calls[name].append(ms)

    def elapsed_ms(self):
        return int((time.perf_counter() - self.t0) * 1000)

    def model_runtimes(self):
        out = []
        for name, xs in self.calls.items():
            xs = sorted(xs)
            k = len(xs)
            out.append({
                "model_name": name,
                "call_count": k,
                "total_time_ms": round(sum(xs), 2),
                "average_time_ms": round(sum(xs) / k, 2),
                "p50_time_ms": round(xs[int(0.50 * (k - 1))], 2),
                "p95_time_ms": round(xs[int(0.95 * (k - 1))], 2),
                "max_time_ms": round(xs[-1], 2),
            })
        return out

    def metadata(self):
        return {
            "frames_processed": self.frames,
            "chunks_processed": self.chunks,
            "end_to_end_internal_time_ms": self.elapsed_ms(),
            "model_runtimes": self.model_runtimes(),
        }


class _Span:
    def __init__(self, timer, name):
        self.timer, self.name = timer, name

    def __enter__(self):
        self.s = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.timer.record(self.name, (time.perf_counter() - self.s) * 1000.0)
        return False
