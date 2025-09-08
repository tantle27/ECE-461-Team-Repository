from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


class UrlType(Enum):
    """High-level categories used to route URLs to handlers and decide output behavior."""
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
        hf_id: Hugging Face repo or dataset ID (e.g., 'org/name'), if applicable.
        gh_owner_repo: ('owner','repo') for GitHub URLs, if applicable.
    """
    raw: str
    type: UrlType
    hf_id: Optional[str] = None
    gh_owner_repo: Optional[Tuple[str, str]] = None


class UrlRouter:
    """
    Classifies input URLs and extracts useful IDs for handlers.

    Responsibilities:
        - Decide UrlType (MODEL, DATASET, CODE, UNKNOWN).
        - Extract identifiers 
    """

    # Hugging Face patterns:
    #   - Model:   https://huggingface.co/{org}/{name}[...]
    #   - Dataset: https://huggingface.co/datasets/{org}/{name}[...]
    # GitHub pattern:
    #   https://github.com/{owner}/{repo}[/...]
    _HF_DATASET_RE = re.compile(
        r"^https?://huggingface\.co/datasets/(?P<org>[^/\s\?#]+)/(?P<name>[^/\s\?#]+)",
        re.IGNORECASE,
    )
    _HF_MODEL_RE = re.compile(
        r"^https?://huggingface\.co/(?P<org>[^/\s\?#]+)/(?P<name>[^/\s\?#]+)",
        re.IGNORECASE,
    )
    _GH_RE = re.compile(
        r"^https?://github\.com/(?P<owner>[^/\s\?#]+)/(?P<repo>[^/\s\?#]+)(?:[/?].*)?$",
        re.IGNORECASE,
    )


    def classify(self, url: str) -> UrlType:
        """
        Returns a UrlType based on hostname/path patterns.
        """
        if self._HF_DATASET_RE.match(url):
            return UrlType.DATASET
        if self._HF_MODEL_RE.match(url) and "/datasets/" not in url.lower():
            return UrlType.MODEL
        if self._GH_RE.match(url):
            return UrlType.CODE
        return UrlType.UNKNOWN

    def parse(self, url: str) -> ParsedUrl:
        """
        Parses a URL into a ParsedUrl with type and normalized identifiers.

        """
        # Dataset
        m_ds = self._HF_DATASET_RE.match(url)
        if m_ds:
            org = m_ds.group("org")
            name = m_ds.group("name")
            return ParsedUrl(raw=url, type=UrlType.DATASET, hf_id=f"{org}/{name}")

        # HF model 
        m_model = self._HF_MODEL_RE.match(url)
        if m_model and "/datasets/" not in url.lower():
            org = m_model.group("org")
            name = m_model.group("name")
            return ParsedUrl(raw=url, type=UrlType.MODEL, hf_id=f"{org}/{name}")

        # GitHub
        m_gh = self._GH_RE.match(url)
        if m_gh:
            owner = m_gh.group("owner")
            repo = m_gh.group("repo")
            return ParsedUrl(raw=url, type=UrlType.CODE, gh_owner_repo=(owner, repo))

        # Fallback
        return ParsedUrl(raw=url, type=UrlType.UNKNOWN)

