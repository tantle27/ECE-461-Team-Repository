import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


class UrlType(Enum):
    """
    High-level categories used to route URLs to handlers and decide output
    behavior.
    """
    MODEL = auto()
    DATASET = auto()
    CODE = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class ParsedUrl:
    """
    Structured info extracted from a URL to help handlers.

    Attributes:
        raw: The original URL string.
        type: One of UrlType.
        hf_id: Hugging Face repo or dataset ID (e.g., 'org/name'), if
            applicable.
        gh_owner_repo: ('owner', 'repo') for GitHub URLs, if applicable.
    """
    raw: str
    type: UrlType
    hf_id: Optional[str] = None
    gh_owner_repo: Optional[Tuple[str, str]] = None


class UrlRouter:
    """Classifies URLs and extracts normalized identifiers."""

    _HF_DATASET_RE = re.compile(
        (
            r"^https?://huggingface\.co/datasets/"
            r"(?P<org>[^/\s]+)/(?P<name>[^/\s]+)"
        ),
        re.IGNORECASE,
    )

    _HF_MODEL_RE = re.compile(
        (
            r"^https?://huggingface\.co/"
            r"(?P<org>[^/\s]+)/(?P<name>[^/\s]+)"
        ),
        re.IGNORECASE,
    )

    _GH_RE = re.compile(
        (
            r"^https?://github\.com/"
            r"(?P<owner>[^/\s]+)/(?P<repo>[^/\s]+)"
            r"(?:/.*)?$"
        ),
        re.IGNORECASE,
    )

    @staticmethod
    def strip_query(url: str) -> str:
        """Remove query string and fragment from a URL."""
        q = url.split("?", 1)[0]
        return q.split("#", 1)[0]

    def classify(self, url: str) -> UrlType:
        """Return only the UrlType (delegates to parse)."""
        return self.parse(url).type

    def parse(self, url: str) -> ParsedUrl:
        """Classify and extract identifiers, normalized to lowercase."""
        clean = self.strip_query(url)
        low = clean.lower()

        m = self._HF_DATASET_RE.match(clean)
        if m:
            org = m.group("org").lower()
            name = m.group("name").lower()
            return ParsedUrl(
                raw=url,
                type=UrlType.DATASET,
                hf_id=f"{org}/{name}",
            )

        m = self._HF_MODEL_RE.match(clean)
        if m and "/datasets/" not in low and "/spaces/" not in low:
            org = m.group("org").lower()
            name = m.group("name").lower()
            return ParsedUrl(
                raw=url,
                type=UrlType.MODEL,
                hf_id=f"{org}/{name}",
            )

        m = self._GH_RE.match(clean)
        if m:
            owner = m.group("owner").lower()
            repo = m.group("repo").lower()
            return ParsedUrl(
                raw=url,
                type=UrlType.CODE,
                gh_owner_repo=(owner, repo),
            )

        return ParsedUrl(raw=url, type=UrlType.UNKNOWN)

    def parse_hf_model_id(self, url: str) -> Optional[str]:
        """Return 'org/name' if URL is a HF model; else None."""
        parsed = self.parse(url)
        return parsed.hf_id if parsed.type is UrlType.MODEL else None

    def parse_hf_dataset_id(self, url: str) -> Optional[str]:
        """Return 'org/name' if URL is a HF dataset; else None."""
        parsed = self.parse(url)
        return parsed.hf_id if parsed.type is UrlType.DATASET else None

    def parse_github_owner_repo(
        self,
        url: str,
    ) -> Optional[Tuple[str, str]]:
        """Return (owner, repo) if URL is a GitHub repo; else None."""
        parsed = self.parse(url)
        return parsed.gh_owner_repo if parsed.type is UrlType.CODE else None
