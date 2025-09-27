from __future__ import annotations

import sys
import os
import random
import re
import time
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional
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

    def __init__(self, *args, timeout: int = DEFAULT_TIMEOUT, **kwargs) -> None:
        self._timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
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


_GITHUB_REPO_RE = re.compile(r"https?://github\.com/([-.\w]+)/([-.\w]+)", re.I)


def normalize_and_verify_github(
    gh: "GHClient",
    urls: Iterable[str],
) -> list[str]:
    valid: list[str] = []
    for u in urls:
        m = _GITHUB_REPO_RE.match(u or "")
        if not m:
            continue
        owner, repo = m.group(1), m.group(2)
        info = gh.get_repo(owner, repo)
        if info is not None:
            valid.append(f"https://github.com/{owner}/{repo}")
    return list(dict.fromkeys(valid))


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
    remaining, reset = resp.headers.get("X-RateLimit-Remaining"), resp.headers.get(
        "X-RateLimit-Reset"
    )
    if remaining == "0" and reset is not None:
        try:
            delay = max(0, int(reset) - int(time.time())) + random.uniform(0.25, 0.75)
            delay = min(delay, 60.0)  # cap long sleeps
            logging.info(
                "GitHub rate limit reached. Sleeping ~%.1fs (reset=%s)",
                delay,
                reset,
            )
            time.sleep(delay)
            return
        except ValueError:
            pass

    logging.info("GitHub throttling/backoff. Sleeping 2s.")
    time.sleep(2.0)


def _etag_key(url: str) -> str:
    return f"ETAG::{url}"


# ---------------- client ----------------


class GHClient:
    """Lightweight GitHub client with ETag and basic rate-limit awareness."""

    def __init__(self) -> None:
        token = os.getenv("GITHUB_TOKEN")
        if not token or not self._is_token_valid(token):
            logging.error(
                "GHClient initialization failed: GITHUB_TOKEN is missing or invalid format."
            )
            # print("Error: GITHUB_TOKEN is missing or invalid format.", file=sys.stderr)
            sys.exit(1)
        self._http = _make_session(token)
        # Check token validity by making a call to GitHub API
        try:
            resp = self._http.get("https://api.github.com/user")
            if resp.status_code != 200:
                logging.error(
                    "GHClient initialization failed: GITHUB_TOKEN is not valid (status=%d).",
                    resp.status_code,
                )
                # print("Error: GITHUB_TOKEN is not valid.", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            logging.error(
                "GHClient initialization failed: Exception during token check: %s", e
            )
            # print("Error: Could not verify GITHUB_TOKEN.", file=sys.stderr)
            sys.exit(1)
        self._etag_cache: dict[str, str] = {}
        logging.debug(
            "GHClient initialized (token=%s)", "present" if token else "absent"
        )

    @staticmethod
    def _is_token_valid(token: str) -> bool:
        # Basic check: GitHub tokens usually start with 'ghp_' and are 40+ chars
        return token.startswith("ghp_") and len(token) >= 40

    # -------- public --------

    def get_repo(self, owner: str, repo: str) -> GHRepoInfo | None:
        data = self._get_json(f"/repos/{owner}/{repo}")
        if data is None:
            logging.debug("get_repo: %s/%s -> not found or not modified", owner, repo)
            return None
        logging.debug("get_repo: %s/%s -> ok", owner, repo)
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
            logging.debug("get_readme_markdown: no readme for %s/%s", owner, repo)
            return None
        url = data["download_url"]
        logging.debug("get_readme_markdown: downloading raw readme %s", url)
        return self._get_text_absolute(url)

    def list_contributors(
        self,
        owner: str,
        repo: str,
        *,
        max_pages: int = 3,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page = 1
        while page <= max_pages:
            logging.debug("list_contributors: page %d for %s/%s", page, owner, repo)
            batch = self._get_json(
                f"/repos/{owner}/{repo}/contributors?per_page=100&page={page}"
            )
            if not batch:
                break
            items.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        logging.debug("list_contributors: total=%d for %s/%s", len(items), owner, repo)
        return items

    # -------- internals --------

    def _github_get(self, url: str, *, use_etag: bool = True) -> requests.Response:
        headers: dict[str, str] = {}
        if use_etag:
            etag = self._etag_cache.get(_etag_key(url))
            if etag:
                headers["If-None-Match"] = etag
                logging.debug("GET %s (If-None-Match sent)", url)
            else:
                logging.debug("GET %s", url)
        else:
            logging.debug("GET %s (no ETag)", url)

        try:
            resp = self._http.get(url, headers=headers)
        except Exception as e:
            logging.error("Request failed: %s (%s)", url, e)
            raise

        if resp.status_code in (403, 429):
            _sleep_until_reset(resp)
            resp = self._http.get(url, headers=headers)

        if resp.status_code == 200:
            new_etag = resp.headers.get("ETag")
            if new_etag:
                self._etag_cache[_etag_key(url)] = new_etag
                logging.debug("Cached ETag for %s", url)
        elif resp.status_code == 304:
            logging.debug("304 Not Modified for %s", url)
        elif resp.status_code == 404:
            logging.debug("404 Not Found for %s", url)
        elif resp.status_code >= 400:
            logging.warning("GitHub responded %d for %s", resp.status_code, url)

        return resp

    def _get_text_absolute(self, url: str) -> str | None:
        resp = self._github_get(url, use_etag=True)
        if resp.status_code in (304, 404):
            return None
        resp.raise_for_status()
        return resp.text

    def _get_json(self, path: str) -> dict | list | None:
        url = urljoin("https://api.github.com", path)
        resp = self._github_get(url, use_etag=True)
        if resp.status_code in (304, 404):
            return None
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError as e:
            logging.warning("Failed to parse JSON from %s: %s", url, e)
            return None
