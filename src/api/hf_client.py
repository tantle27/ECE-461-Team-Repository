import os
import time
from dataclasses import dataclass
from typing import Any, Optional
from collections.abc import Mapping

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
    adapter = _TimeoutHTTPAdapter(max_retries=_retry_policy(), timeout=30)
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


def _normalize_card_data(card_obj: Any) -> dict[str, Any]:
    """
    Convert huggingface_hub ModelCardData (or similar) to a plain dict.
    Works across hub versions; falls back to {} if anything goes sideways.
    """
    try:
        if card_obj is None:
            return {}
        # Most versions expose .to_dict()
        if hasattr(card_obj, "to_dict") and callable(card_obj.to_dict):
            return dict(card_obj.to_dict())
        # Some expose .data (already a mapping)
        if hasattr(card_obj, "data"):
            data = getattr(card_obj, "data")
            if isinstance(data, Mapping):
                return dict(data)
        # Already a dict?
        if isinstance(card_obj, Mapping):
            return dict(card_obj)
        # Pydantic-like or dataclass: try .model_dump() / .dict() / .json()
        if hasattr(card_obj, "model_dump"):
            return dict(card_obj.model_dump())
        if hasattr(card_obj, "dict"):
            return dict(card_obj.dict())
        if hasattr(card_obj, "json"):
            import json
            return json.loads(card_obj.json())
    except Exception:
        pass
    return {}
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
        # Accept any of the common env var names
        self.token = (
            os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HF_TOKEN")
        )
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
            paths = _retry(lambda: self.api.list_repo_files(
                hf_id, repo_type=repo_type))
            try:
                infos = _retry(lambda: self.api.get_paths_info(
                    hf_id, paths=paths, repo_type=repo_type))
                return [HFFileInfo(path=i.path, size=i.size) for i in infos]
            except AttributeError:
                return [HFFileInfo(path=p, size=None) for p in paths]
        except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError):
            raise

    def get_readme(
        self,
        hf_id: str,
        *,
        revision: str = "main",
        repo_type: Optional[str] = None,
    ) -> str | None:
        """
        Try to fetch README text for models or datasets.
        Accepts hf_id in either form: 'org/name' or 'datasets/org/name'.
        """
        # Normalize both forms
        plain = hf_id.removeprefix("datasets/")
        with_prefix = f"datasets/{plain}"

        bases: list[str]
        if repo_type == "dataset":
            bases = [with_prefix, plain]
        elif repo_type == "model":
            bases = [plain, with_prefix]
        else:
            bases = [plain, with_prefix]

        candidates: list[str] = []
        for base in bases:
            candidates.extend([
                f"https://huggingface.co/{base}/raw/{revision}/README.md",
                f"https://huggingface.co/{base}/raw/{revision}/README.MD",
                f"https://huggingface.co/{base}/raw/{revision}/readme.md",
            ])

        for u in candidates:
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
        # Try both forms here as well, just in case
        plain = hf_id.removeprefix("datasets/")
        with_prefix = f"datasets/{plain}"
        urls = [
            f"https://huggingface.co/{plain}/raw/{revision}/model_index.json",
            f"https://huggingface.co/{
                with_prefix}/raw/{revision}/model_index.json",
        ]
        for url in urls:
            data = _retry(lambda: self._get_json(url))
            if data is not None:
                return data
        return None

    # ---- internals ----

    def _to_info(self, hf_id: str, info: Any) -> HFModelInfo:
        # Normalize card data safely
        card_data = _normalize_card_data(getattr(info, "cardData", None))

        # Normalize tags → list[str]
        raw_tags = getattr(info, "tags", None)
        if raw_tags is None:
            tags: list[str] = []
        elif isinstance(raw_tags, (list, set, tuple)):
            tags = [str(t) for t in raw_tags]
        else:
            tags = [str(raw_tags)]

        return HFModelInfo(
            hf_id=hf_id,
            card_data=card_data,
            tags=tags,
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
        if r.status_code == 401:
            # Surface as gated/private so handlers can set ctx.gated=True
            raise GatedRepoError(
                f"401 Unauthorized for {url}. Token missing or access not granted.")
        r.raise_for_status()
        return r.text

    def _get_json(self, url: str) -> dict | None:
        r = self._http.get(url)
        if r.status_code == 404:
            return None
        if r.status_code == 401:
            raise GatedRepoError(f"401 Unauthorized for {url}. Token missing or access not granted.")
        r.raise_for_status()
        return r.json()