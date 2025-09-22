import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from huggingface_hub import HfApi
from huggingface_hub.utils import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)


class _TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, *args, timeout: int = 30, **kwargs):
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
                                  timeout=30)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    headers = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "acme-model-evaluator/0.1 (+requests)",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    s.headers.update(headers)
    return s


# ---- data models ----

@dataclass
class HFModelInfo:
    hf_id: str
    card_data: dict[str, Any] | None
    tags: list[str]
    likes: int | None
    downloads_30d: int | None
    downloads_all_time: int | None
    created_at: str | None
    last_modified: str | None
    gated: bool | None
    private: bool | None


@dataclass
class HFFileInfo:
    path: str
    size: int | None


# ---- util retry wrapper ----

def _retry(fn, attempts: int = 4):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            if i == attempts - 1:
                raise e
            time.sleep(0.4 * (2**i))
    return None


# ---- client ----

class HFClient:
    """Convenience wrapper around huggingface_hub + raw HTTP (requests)."""

    def __init__(self) -> None:
        self.token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.api = HfApi(token=self.token)
        self._http = _make_session(self.token)

    # ---- public ----

    def get_model_info(self, hf_id: str) -> HFModelInfo:
        info = _retry(lambda: self.api.model_info(hf_id))
        return self._to_info(hf_id, info)

    def get_dataset_info(self, hf_id: str) -> HFModelInfo:
        info = _retry(lambda: self.api.dataset_info(hf_id))
        return self._to_info(hf_id, info)

    def list_files(
        self,
        hf_id: str,
        *,
        repo_type: str = "model",
    ) -> list[HFFileInfo]:
        """repo_type in {'model','dataset'}."""
        try:
            paths = _retry(
                lambda: self.api.list_repo_files(hf_id, repo_type=repo_type)
            )
            try:
                infos = _retry(
                    lambda: self.api.get_paths_info(
                        hf_id, paths=paths, repo_type=repo_type
                    )
                )
                return [HFFileInfo(path=i.path, size=i.size) for i in infos]
            except AttributeError:
                return [HFFileInfo(path=p, size=None) for p in paths]
        except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError):
            raise

    def get_readme(self, hf_id: str, *, revision: str = "main") -> str | None:
        urls = [
            f"https://huggingface.co/{hf_id}/raw/{revision}/README.md",
            f"https://huggingface.co/{hf_id}/raw/{revision}/README.MD",
            f"https://huggingface.co/{hf_id}/raw/{revision}/readme.md",
        ]
        for u in urls:
            text = _retry(lambda: self._get_text(u))
            if text:
                return text
        return None

    def get_model_index_json(
        self,
        hf_id: str,
        *,
        revision: str = "main",
    ) -> dict | None:
        url = f"https://huggingface.co/{hf_id}/raw/{revision}/model_index.json"
        return _retry(lambda: self._get_json(url))

    # ---- internals ----

    def _to_info(self, hf_id: str, info: Any) -> HFModelInfo:
        return HFModelInfo(
            hf_id=hf_id,
            card_data=getattr(info, "cardData", None) or {},
            tags=list(getattr(info, "tags", []) or []),
            likes=getattr(info, "likes", None),
            downloads_30d=getattr(info, "downloads", None),
            downloads_all_time=getattr(info, "downloadsAllTime", None),
            created_at=str(getattr(info, "created_at", None)),
            last_modified=str(getattr(info, "last_modified", None)),
            gated=getattr(info, "gated", None),
            private=getattr(info, "private", None),
        )

    def _get_text(self, url: str) -> str | None:
        r = self._http.get(url)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text

    def _get_json(self, url: str) -> dict | None:
        r = self._http.get(url)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
