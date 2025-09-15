import os
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class GHRepoInfo:
    owner: str
    repo: str
    private: bool | None
    default_branch: str | None
    description: str | None


def _retry(fn, attempts: int = 4):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise e
            time.sleep(0.4 * (2**i))
    return None


DEFAULT_TIMEOUT = 30  # seconds


class _TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, *args, timeout: int = DEFAULT_TIMEOUT, **kwargs):
        self._timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):  # type: ignore[override]
        kwargs.setdefault("timeout", self._timeout)
        return super().send(request, **kwargs)


def _retry_policy() -> Retry:
    return Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )


def _make_session(token: Optional[str]) -> requests.Session:
    s = requests.Session()
    adapter = _TimeoutHTTPAdapter(max_retries=_retry_policy(),
                                  timeout=DEFAULT_TIMEOUT)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "acme-model-evaluator/0.1 (+requests)",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    s.headers.update(headers)
    return s


class GHClient:
    def __init__(self) -> None:
        token = os.getenv("GITHUB_TOKEN")
        self._http = _make_session(token)

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

    def list_contributors(self, owner: str, repo: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page = 1
        while True:
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

    def _get_text_absolute(self, url: str) -> str | None:
        def _go():
            r = self._http.get(url)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text

        return _retry(_go)

    def _get_json(self, path: str) -> dict | list | None:
        def _go():
            url = urljoin("https://api.github.com", path)
            r = self._http.get(url)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()

        return _retry(_go)
