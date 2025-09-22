import os
import random
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------- data models ----------------


@dataclass
class GHRepoInfo:
    owner: str
    repo: str
    private: bool | None
    default_branch: str | None
    description: str | None


DEFAULT_TIMEOUT = 30  # seconds


# ---------------- retry + session ----------------


class _TimeoutHTTPAdapter(HTTPAdapter):
    """HTTPAdapter that applies a default timeout and retry policy."""

    def __init__(self, *args, timeout: int = DEFAULT_TIMEOUT,
                 **kwargs) -> None:
        self._timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):  # type: ignore[override]
        kwargs.setdefault("timeout", self._timeout)
        return super().send(request, **kwargs)


def _retry_policy() -> Retry:
    """Retry on transient server/network issues."""
    return Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )


def _make_session(token: Optional[str]) -> requests.Session:
    """Build a requests session with headers, timeout, and retry policy."""
    session = requests.Session()
    adapter = _TimeoutHTTPAdapter(
        max_retries=_retry_policy(),
        timeout=DEFAULT_TIMEOUT,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "acme-model-evaluator/0.1 (+requests)",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session.headers.update(headers)
    return session


# ---------------- rate limit helpers ----------------


def _sleep_until_reset(resp: requests.Response) -> None:
    remaining, reset = resp.headers.get(
        "X-RateLimit-Remaining"), resp.headers.get("X-RateLimit-Reset")
    if remaining == "0" and reset is not None:
        try:
            delay = max(0, int(reset) - int(
                time.time())) + random.uniform(0.25, 0.75)
            time.sleep(min(delay, 60.0))  # cap long sleeps
            return
        except ValueError:
            pass

    time.sleep(2.0)


def _etag_key(url: str) -> str:
    return f"ETAG::{url}"


# ---------------- client ----------------


class GHClient:
    """Lightweight GitHub client with ETag and basic rate-limit awareness."""

    def __init__(self) -> None:
        token = os.getenv("GITHUB_TOKEN")
        self._http = _make_session(token)
        # Tiny in-memory ETag cache; swap with persistent storage if needed.
        self._etag_cache: dict[str, str] = {}

    # -------- public --------

    def get_repo(self, owner: str, repo: str) -> GHRepoInfo | None:
        data = self._get_json(f"/repos/{owner}/{repo}")
        if data is None:
            return None
        return GHRepoInfo(
            owner=owner,
            repo=repo,
            private=data.get("private"),
            default_branch=data.get("default_branch"),
            description=data.get("description"),
        )

    def get_readme_markdown(self, owner: str, repo: str) -> str | None:
        data = self._get_json(f"/repos/{owner}/{repo}/readme")
        if not data or "download_url" not in data:
            return None
        return self._get_text_absolute(data["download_url"])

    def list_contributors(
        self,
        owner: str,
        repo: str,
        *,
        max_pages: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Paginate conservatively to avoid burning quota.
        max_pages=3 → up to 300 contributors.
        """
        items: list[dict[str, Any]] = []
        page = 1
        while page <= max_pages:
            batch = self._get_json(
                f"/repos/{owner}/{repo}/contributors?per_page=100&page={page}"
            )
            if not batch:
                break
            items.extend(batch)  # type: ignore[arg-type]
            if len(batch) < 100:  # type: ignore[arg-type]
                break
            page += 1
        return items

    # -------- internals --------

    def _github_get(self, url: str, *, use_etag: bool
                    = True) -> requests.Response:
        headers: dict[str, str] = {}
        if use_etag:
            etag = self._etag_cache.get(_etag_key(url))
            if etag:
                headers["If-None-Match"] = etag

        resp = self._http.get(url, headers=headers)

        if resp.status_code in (403, 429):
            _sleep_until_reset(resp)
            resp = self._http.get(url, headers=headers)

        # Cache ETag on success
        if resp.status_code == 200:
            new_etag = resp.headers.get("ETag")
            if new_etag:
                self._etag_cache[_etag_key(url)] = new_etag

        return resp

    def _get_text_absolute(self, url: str) -> str | None:
        # For raw.githubusercontent URLs and readme download_url
        resp = self._github_get(url, use_etag=True)
        if resp.status_code == 304:
            return None  # unchanged since last time
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text

    def _get_json(self, path: str) -> dict | list | None:
        url = urljoin("https://api.github.com", path)
        resp = self._github_get(url, use_etag=True)
        if resp.status_code == 304:
            return None  # unchanged; skip work
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
