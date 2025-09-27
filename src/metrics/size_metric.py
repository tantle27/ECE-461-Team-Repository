# metrics/size_metric.py
from __future__ import annotations
from .base_metric import BaseMetric
from repo_context import RepoContext


def _clamp01(x: float) -> float:
    if x < 0.0: 
        return 0.0
    if x > 1.0: 
        return 1.0
    return x


class SizeMetric(BaseMetric):
    """
    Device-aware size metric.

    Per-device score ~ 1 - (size_gb / T_device), clamped to [0,1].
    T_device is the rough "comfortable" capacity for that device class.
    The scalar Size score we return is the Desktop score (acts as a good
    overall deployment proxy and matches your expected outputs).
    """

    def __init__(self, weight: float = 0.10):
        super().__init__(name="Size", weight=weight)
        self.T_RPI = 0.50   # ~512 MB
        self.T_NANO = 0.67   # ~672 MB
        self.T_DESKTOP = 8.00   # ~8  GB
        self.T_AWS = 16.00  # ~16 GB

    def _gb(self, bytes_val: int) -> float:
        # decimal GB (10^9) — matches HF sizes and your tests
        return float(bytes_val) / (1000.0 ** 3)

    def _score_cap(self, size_gb: float, cap_gb: float) -> float:
        # 1 - (size / cap), clamped
        if cap_gb <= 0.0:
            return 0.0
        return _clamp01(1.0 - (size_gb / cap_gb))

    def _compute_bytes(self, repo_context: dict) -> int:
        # Prefer explicit file sizes
        files = repo_context.get("files") or []
        total = 0
        try:
            for f in files:
                total += int(getattr(f, "size_bytes", 0) or 0)
        except Exception:
            total = 0

        # Fallback to RepoContext’s known-weight sum
        if total <= 0:
            ctx = repo_context.get("_ctx_obj")
            if isinstance(ctx, RepoContext):
                try:
                    total = int(ctx.total_weight_bytes())
                except Exception:
                    total = 0
        return max(0, total)

    def evaluate(self, repo_context: dict) -> float:
        size_bytes = self._compute_bytes(repo_context)
        size_gb = self._gb(size_bytes)

        # Per-device scores
        rpi = self._score_cap(size_gb, self.T_RPI)
        nano = self._score_cap(size_gb, self.T_NANO)
        desk = self._score_cap(size_gb, self.T_DESKTOP)
        aws = self._score_cap(size_gb, self.T_AWS)

        # Expose per-device scores for NDJSON emission
        ctx = repo_context.get("_ctx_obj")
        if isinstance(ctx, RepoContext):
            ctx.__dict__["_size_device_scores"] = {
                "raspberry_pi": round(rpi, 2),
                "jetson_nano": round(nano, 2),
                "desktop_pc":  round(desk, 2),
                "aws_server":  round(aws, 2),
            }

        # Scalar Size score = Desktop score (matches your expected reference)
        return float(desk)

    def get_description(self) -> str:
        return "Evaluates model size impact on device usability via per-device caps"
