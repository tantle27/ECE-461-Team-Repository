from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable, Optional

from api.gh_client import normalize_and_verify_github
from api.hf_client import github_urls_from_readme
from url_router import UrlRouter, UrlType
import re


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
        return self.total_weight_bytes() / (1000**3)

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
            self.files.append(FileInfo(path=p, size_bytes=0, ext=ext))

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

    # inside RepoContext
    def hydrate_code_links(self, hf_client, gh_client) -> None:
        if self.gh_url:
            return
        url = find_code_repo_url(
            hf_client, gh_client, self, prefer_readme=True
        )
        if not url:
            return
        self.gh_url = url
        self.link_code(RepoContext(url=url, gh_url=url, host="github.com"))


def find_code_repo_url(
    hf_client, gh_client, ctx: RepoContext, *, prefer_readme: bool = True
) -> Optional[str]:
    candidates: list[str] = []
    if ctx.readme_text and ctx.hf_id:
        candidates.extend(
            github_urls_from_readme(
                ctx.hf_id, ctx.readme_text, card_data=ctx.card_data
            )
        )

    if not candidates and ctx.hf_id:
        p = ctx.hf_id.replace("datasets/", "")
        if "/" in p:
            org, name = p.split("/", 1)
            guess = f"https://github.com/{org}/{_norm(name)}".replace(
                "--", "-"
            )
            candidates.append(guess)

    verified = normalize_and_verify_github(gh_client, candidates)

    if not verified:
        return None

    # prefer README-derived one
    if prefer_readme and ctx.readme_text and ctx.hf_id:
        for u in github_urls_from_readme(
            ctx.hf_id, ctx.readme_text, card_data=ctx.card_data
        ):
            if u in verified:
                return u

    # tie-break: prefer same owner + best version
    p = ctx.hf_id.replace("datasets/", "")
    hf_org = p.split("/", 1)[0].lower() if "/" in p else p.lower()

    def ver_score(u: str) -> float:
        repo = u.rsplit("/", 1)[-1]
        return _ver_bonus(ctx.hf_id, repo)

    same_owner = [u for u in verified if f"/{hf_org}/" in u.lower()]
    if same_owner:
        same_owner.sort(key=ver_score, reverse=True)
        return same_owner[0]

    verified.sort(key=ver_score, reverse=True)
    return verified[0]


_VERSION_RE = re.compile(r"(?<![a-z0-9])v?(\d+(?:\.\d+)*)(?![a-z0-9])", re.I)


def _norm(s: str) -> str:
    """Normalize a name for GH repo guessing (lower + simple cleanup)."""
    s = (s or "").lower()
    for pre in ("hf-", "huggingface-", "the-"):
        if s.startswith(pre):
            s = s[len(pre) :]
    for suf in (
        "-dev",
        "-devkit",
        "-main",
        "-release",
        "-project",
        "-repo",
        "-code",
    ):
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def _ver_bonus(hf_id: str | None, repo_name: str | None) -> float:
    """
    Small score bonus based on version similarity between HF id and GH repo
    name. Exact match → +0.30, off-by-one segment → +0.10, otherwise -0.20.
    """

    def parse_vers(txt: str | None) -> list[tuple[int, ...]]:
        if not txt:
            return []
        return [
            tuple(int(x) for x in m.group(1).split("."))
            for m in _VERSION_RE.finditer(txt)
        ]

    hv = parse_vers(hf_id)
    rv = parse_vers(repo_name)
    if not hv or not rv:
        return 0.0

    def dist(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        n = max(len(a), len(b))
        ap = a + (0,) * (n - len(a))
        bp = b + (0,) * (n - len(b))
        return sum(1 for i in range(n) if ap[i] != bp[i])

    m = min(dist(a, b) for a in hv for b in rv)
    if m == 0:
        return 0.30
    if m == 1:
        return 0.10
    return -0.20
