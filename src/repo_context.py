from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Optional

from url_router import UrlRouter, UrlType


@dataclass(frozen=True)
class FileInfo:
    """Lightweight descriptor for a file in the local checkout."""
    path: Path
    size_bytes: int
    ext: str


@dataclass
class RepoContext:
    """
    Mutable bundle of facts used by metrics and handlers.
    Handlers populate fields; metrics treat as read-only.
    """

    # Source
    url: Optional[str] = None
    hf_id: Optional[str] = None
    gh_url: Optional[str] = None
    host: Optional[str] = None

    # Local checkout
    repo_path: Optional[Path] = None
    files: list[FileInfo] = field(default_factory=list)

    # Extracted metadata
    readme_text: Optional[str] = None
    config_json: Optional[dict[str, Any]] = None
    model_index: Optional[dict[str, Any]] = None

    # Hugging Face metadata
    card_data: Optional[dict[str, Any]] = None
    tags: list[str] = field(default_factory=list)
    downloads_30d: Optional[int] = None
    downloads_all_time: Optional[int] = None
    likes: Optional[int] = None
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    gated: Optional[bool] = None
    private: Optional[bool] = None

    # GitHub metadata
    contributors: list[dict[str, Any]] = field(default_factory=list)
    commit_history: list[dict[str, Any]] = field(default_factory=list)

    # Linkage (for models)
    linked_datasets: list["RepoContext"] = field(default_factory=list)
    linked_code: list["RepoContext"] = field(default_factory=list)

    # Diagnostics
    fetch_logs: list[str] = field(default_factory=list)
    cache_hits: int = 0
    api_errors: int = 0

    # Known weight file extensions
    WEIGHT_EXTENSIONS: Final[tuple[str, ...]] = (
        "safetensors",
        "bin",
        "pt",
        "pth",
        "h5",
        "onnx",
        "tflite",
        "gguf",
        "ggml",
        "ckpt",
    )

    @classmethod
    def _canon_dataset_key(cls, ctx: "RepoContext") -> str:
        """
        Canonical key for a dataset context.

        Prefer normalized hf_id ('org/name'); else parse from URL; else URL.
        """
        if ctx.hf_id:
            return ctx.hf_id.lower().removeprefix("datasets/")
        if ctx.url:
            parsed = UrlRouter().parse(ctx.url)
            if parsed.type is UrlType.DATASET and parsed.hf_id:
                return parsed.hf_id
            return ctx.url.lower()
        return ""

    @classmethod
    def _canon_code_key(cls, ctx: "RepoContext") -> str:
        """
        Canonical key for a code repo.

        Normalize GitHub URLs to 'https://github.com/owner/repo'.
        """
        if ctx.gh_url:
            parsed = UrlRouter().parse(ctx.gh_url)
            if parsed.type is UrlType.CODE and parsed.gh_owner_repo:
                owner, repo = parsed.gh_owner_repo
                return f"https://github.com/{owner}/{repo}"
            return ctx.gh_url.lower()

        if ctx.url:
            parsed = UrlRouter().parse(ctx.url)
            if parsed.type is UrlType.CODE and parsed.gh_owner_repo:
                owner, repo = parsed.gh_owner_repo
                return f"https://github.com/{owner}/{repo}"
            return ctx.url.lower()

        return ""

    def total_weight_bytes(self) -> int:
        """Sum sizes of known weight files (bytes)."""
        total = 0
        for fi in self.files:
            if fi.ext.lower() in self.WEIGHT_EXTENSIONS:
                total += max(0, fi.size_bytes)
        return total

    def total_weight_gb(self) -> float:
        """Total weight size in GiB."""
        return self.total_weight_bytes() / (1024**3)

    def add_files(self, paths: Iterable[Path]) -> None:
        """
        Register files without stat-ing.

        Sizes remain 0 unless set by handlers.
        """
        seen = {fi.path for fi in self.files}
        for p in paths:
            if p in seen:
                continue
            ext = (
                p.suffix[1:].lower()
                if p.suffix.startswith(".")
                else p.suffix.lower()
            )
            self.files.append(
                FileInfo(path=p, size_bytes=0, ext=ext)
            )

    def link_dataset(self, ds_ctx: "RepoContext") -> None:
        """Associate a dataset context to this context (usually a model)."""
        key = self._canon_dataset_key(ds_ctx)
        if not key:
            return
        if any(
            self._canon_dataset_key(c) == key for c in self.linked_datasets
        ):
            return
        self.linked_datasets.append(ds_ctx)

    def link_code(self, code_ctx: "RepoContext") -> None:
        """Associate a code repo context to this context (usually a model)."""
        key = self._canon_code_key(code_ctx)
        if not key:
            return
        if any(self._canon_code_key(c) == key for c in self.linked_code):
            return
        self.linked_code.append(code_ctx)