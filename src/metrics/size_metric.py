import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

from dulwich import porcelain
from dulwich.client import get_transport_and_path_from_url
from dulwich.errors import NotGitRepository

from .base_metric import BaseMetric
from repo_context import RepoContext


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


_LFS_PTR_RE = re.compile(
    r"^\s*version\s+https?://git-lfs\.github\.com/spec/v1\s*$|^\s*oid\s+sha256:[0-9a-f]{64}\s*$|^\s*size\s+(\d+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_WEIGHT_EXTS = {
    ".bin", ".safetensors", ".h5", ".msgpack", ".ckpt", ".pth", ".pt",
    ".tflite", ".onnx", ".gguf", ".ggml", ".npz",
}
_WEIGHT_NAME_HINTS = {
    "pytorch_model", "tf_model", "flax_model", "model.safetensors",
    "model-00001-of", "consolidated", "model.ckpt",
}


class _no_gitconfigs:
    def __enter__(self):
        self._old_home = os.environ.get("HOME")
        self._tmp_home = tempfile.mkdtemp(prefix="nogitcfg-")
        os.environ["HOME"] = self._tmp_home
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = self._old_home
        finally:
            shutil.rmtree(self._tmp_home, ignore_errors=True)


class SizeMetric(BaseMetric):
    def __init__(self, weight: float = 0.10):
        super().__init__(name="Size", weight=weight)
        self.T_RPI = 0.50
        self.T_NANO = 0.67
        self.T_DESKTOP = 8.00
        self.T_AWS = 16.00
        self.G_RPI = 0.86
        self.G_NANO = 0.30
        self.G_DESKTOP = 2.50
        self.G_AWS = 2.50

    def evaluate(self, repo_context: dict) -> float:
        t0 = time.perf_counter()

        size_bytes = 0
        remote = self._derive_git_remote(repo_context)
        if remote:
            try:
                size_bytes = self._bytes_via_shallow_clone(remote)
            except Exception:
                size_bytes = 0

        if size_bytes <= 0:
            size_bytes = self._bytes_from_context(repo_context)

        size_gb = float(size_bytes) / (1000.0 ** 3)

        rpi = self._soft_cap(size_gb, self.T_RPI, self.G_RPI)
        nano = self._soft_cap(size_gb, self.T_NANO, self.G_NANO)
        desk = self._soft_cap(size_gb, self.T_DESKTOP, self.G_DESKTOP)
        aws = self._soft_cap(size_gb, self.T_AWS, self.G_AWS)

        latency_ms = int(round((time.perf_counter() - t0) * 1000))

        ctx = repo_context.get("_ctx_obj")
        if isinstance(ctx, RepoContext):
            dev_map = {
                "raspberry_pi": round(rpi, 2),
                "jetson_nano": round(nano, 2),
                "desktop_pc": round(desk, 2),
                "aws_server": round(aws, 2),
            }
            # Save directly to the attribute your emitter expects
            ctx.__dict__["_size_device_scores"] = dev_map
            ctx.__dict__["size_score_latency_ms"] = latency_ms

        # print(
        #     f"[SizeMetric] remote={remote} size_bytes={size_bytes} size_gb={size_gb:.3f} "
        #     f"rpi={rpi:.3f} nano={nano:.3f} desk={desk:.3f} aws={aws:.3f} latency_ms={latency_ms}"
        # )
        return float(desk)

    def get_description(self) -> str:
        return "Local clone, LFS-aware size estimate, device-aware scoring."

    def _derive_git_remote(self, repo_context: dict) -> str:
        ctx = repo_context.get("_ctx_obj")
        url = ""
        if isinstance(ctx, RepoContext):
            url = ctx.gh_url or ctx.url or ctx.hf_id or ""
        else:
            url = repo_context.get("url") or repo_context.get("hf_id") or ""
        if not url:
            return ""
        u = self._normalize_repo_url(str(url).strip())
        if not u:
            return ""
        if u.endswith(".git"):
            return u
        return u + ".git"

    def _normalize_repo_url(self, raw: str) -> str:
        if not raw:
            return ""
        if "://" not in raw and "/" in raw and not raw.endswith(".git"):
            return f"https://huggingface.co/{raw}"

        parsed = urlparse(raw)
        host = parsed.netloc.lower()
        path = parsed.path
        parsed = parsed._replace(query="", fragment="")

        if "github.com" in host:
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 2:
                root = f"/{parts[0]}/{parts[1]}"
                return urlunparse(parsed._replace(path=root))
            return urlunparse(parsed)

        if "huggingface.co" in host:
            parts = [p for p in path.split("/") if p]
            if not parts:
                return urlunparse(parsed)
            if parts[0] in {"datasets", "spaces", "models"}:
                if len(parts) >= 3:
                    root = f"/{parts[0]}/{parts[1]}/{parts[2]}"
                    return urlunparse(parsed._replace(path=root))
            if len(parts) >= 2:
                root = f"/{parts[0]}/{parts[1]}"
                return urlunparse(parsed._replace(path=root))
            return urlunparse(parsed)

        return urlunparse(parsed)

    def _bytes_via_shallow_clone(self, remote: str) -> int:
        get_transport_and_path_from_url(remote)
        tmpdir = tempfile.mkdtemp(prefix="size-metric-")
        try:
            with _no_gitconfigs():
                porcelain.clone(remote, tmpdir, checkout=False, depth=1)

            total = 0
            for root, _dirs, files in os.walk(tmpdir):
                if root.endswith(os.sep + ".git"):
                    continue
                for name in files:
                    fp = os.path.join(root, name)
                    try:
                        if not self._is_weight_like(fp):
                            continue
                        lfs_sz = self._maybe_lfs_pointer_size(fp)
                        if lfs_sz is not None:
                            total += max(0, int(lfs_sz))
                            continue
                        st = os.lstat(fp)
                        total += max(0, int(st.st_size))
                    except Exception:
                        continue
            return max(0, int(total))
        except NotGitRepository:
            return 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _maybe_lfs_pointer_size(self, filepath: str) -> Optional[int]:
        try:
            sz = os.path.getsize(filepath)
            if sz > 1024 * 1024:
                return None
            with open(filepath, "rb") as f:
                head = f.read(8192)
            text = head.decode("utf-8", errors="ignore")
            size_val = None
            for m in _LFS_PTR_RE.finditer(text):
                grp = m.groups()
                if grp and grp[0]:
                    try:
                        size_val = int(grp[0])
                    except ValueError:
                        pass
            if size_val is not None:
                return size_val
            if b"git-lfs.github.com/spec/v1" in head:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.lower().startswith("size "):
                            parts = line.strip().split()
                            if len(parts) == 2 and parts[1].isdigit():
                                return int(parts[1])
        except Exception:
            pass
        return None

    def _is_weight_like(self, path: str) -> bool:
        p = Path(path)
        ext = p.suffix.lower()
        if ext in _WEIGHT_EXTS:
            return True
        name = p.name.lower()
        for hint in _WEIGHT_NAME_HINTS:
            if hint in name:
                return True
        if "tokenizer" in name or name.endswith(("config.json", "model_index.json")):
            return False
        return False

    def _bytes_from_context(self, repo_context: dict) -> int:
        files = repo_context.get("files") or []
        total = 0
        try:
            for f in files:
                fn = getattr(f, "name", "") or getattr(f, "path", "")
                if fn and not self._is_weight_like(fn):
                    continue
                total += int(getattr(f, "size_bytes", 0) or 0)
        except Exception:
            total = 0
        if total <= 0:
            ctx = repo_context.get("_ctx_obj")
            if isinstance(ctx, RepoContext):
                try:
                    total = int(ctx.total_weight_bytes())
                except Exception:
                    total = 0
        return max(0, total)

    def _soft_cap(self, size_gb: float, cap_gb: float, gamma: float) -> float:
        if cap_gb <= 0.0:
            return 0.0
        ratio = size_gb / cap_gb if cap_gb > 0 else 0.0
        return _clamp01(1.0 / (1.0 + (ratio ** gamma)))
