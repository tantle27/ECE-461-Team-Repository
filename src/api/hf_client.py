import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import requests
from huggingface_hub import HfApi
from huggingface_hub.utils import (GatedRepoError, HfHubHTTPError,
                                   RepositoryNotFoundError)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class _TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, *args, timeout: int = 30, **kwargs):
        self._timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        kwargs.setdefault("timeout", self._timeout)
        return super().send(request, **kwargs)


def _create_session(token: Optional[str]) -> requests.Session:
    session = requests.Session()
    retry_policy = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = _TimeoutHTTPAdapter(max_retries=retry_policy, timeout=30)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    headers = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "acme-model-evaluator/0.1 (+requests)",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session.headers.update(headers)
    return session


@dataclass
class HFModelInfo:
    hf_id: str
    card_data: Dict[str, Any] | None
    tags: List[str]
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


def _normalize_card_data(card_obj: Any) -> Dict[str, Any]:
    """Convert ModelCardData to dict, handling various hub versions."""
    if card_obj is None:
        return {}
    try:
        if hasattr(card_obj, "to_dict") and callable(card_obj.to_dict):
            return dict(card_obj.to_dict())
        if hasattr(card_obj, "data") and isinstance(card_obj.data, Mapping):
            return dict(card_obj.data)
        if isinstance(card_obj, Mapping):
            return dict(card_obj)
        for method in ("model_dump", "dict", "json"):
            if hasattr(card_obj, method):
                result = getattr(card_obj, method)()
                return json.loads(result) if method == "json" else dict(result)
    except Exception:
        pass
    return {}


def _retry(operation, attempts: int = 4):
    for i in range(attempts):
        try:
            return operation()
        except Exception as e:
            if i == attempts - 1:
                raise e
            time.sleep(0.4 * (2**i))
    return None


# ----------------- GitHub URL matching (unchanged) -----------------


class GitHubMatcher:
    GITHUB_PATTERN = re.compile(r"https?://github\.com/[^\s)\"'<>]+", re.I)
    ROOT_PATTERN = re.compile(r"^(https?://github\.com/[-\w.]+/[-\w.]+)", re.I)
    VERSION_PATTERN = re.compile(
        r"(?:^|[^a-z0-9])v?(\d+(?:\.\d+)*)(?:[^a-z0-9]|$)", re.I
    )
    GENERIC_NAMES = {
        "transformers",
        "diffusers",
        "datasets",
        "examples",
        "tutorials",
        "notebooks",
        "benchmarks",
        "models",
        "scripts",
        "awesome",
        "research",
    }

    @staticmethod
    def _normalize(name: str) -> str:
        name = (name or "").lower()
        for prefix in ("hf-", "huggingface-", "the-"):
            if name.startswith(prefix):
                name = name[len(prefix):]
        for suffix in (
            "-dev",
            "-devkit",
            "-main",
            "-release",
            "-project",
            "-repo",
            "-code",
        ):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return re.sub(r"[^a-z0-9]+", "-", name).strip("-")

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return {
            t
            for t in re.split(r"[^a-z0-9]+", GitHubMatcher._normalize(text))
            if len(t) >= 3
        }

    @staticmethod
    def _get_aliases(hf_id: str, card_data: Dict | None) -> Set[str]:
        aliases = set()
        clean_id = hf_id.replace("datasets/", "") if hf_id else ""

        if "/" in clean_id:
            org, name = clean_id.split("/", 1)
            aliases.update(
                {GitHubMatcher._normalize(name), GitHubMatcher._normalize(org)}
            )
        else:
            aliases.add(GitHubMatcher._normalize(clean_id))

        if card_data:
            model_index = card_data.get("model-index")
            if (
                isinstance(model_index, list)
                and model_index
                and isinstance(model_index[0], dict)
            ):
                name = model_index[0].get("name")
                if isinstance(name, str):
                    aliases.add(GitHubMatcher._normalize(name))

            for field in ("model_name", "name", "title"):
                value = card_data.get(field)
                if isinstance(value, str):
                    aliases.add(GitHubMatcher._normalize(value))

        return {a for a in aliases if a}

    @staticmethod
    def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        return intersection / len(set_a | set_b) if intersection > 0 else 0.0

    @staticmethod
    def _version_bonus(hf_id: str, repo_name: str) -> float:
        hf_versions = [
            tuple(int(x) for x in m.group(1).split("."))
            for m in GitHubMatcher.VERSION_PATTERN.finditer(hf_id or "")
        ]
        repo_versions = [
            tuple(int(x) for x in m.group(1).split("."))
            for m in GitHubMatcher.VERSION_PATTERN.finditer(repo_name or "")
        ]
        if not hf_versions or not repo_versions:
            return 0.0
        return max(
            1.0 if hv == rv else 0.9
            for hv in hf_versions
            for rv in repo_versions
        )

    @classmethod
    def extract_urls(
        cls, hf_id: str, readme: str, card_data: Dict | None = None
    ) -> List[str]:
        if not readme:
            return []

        aliases = cls._get_aliases(hf_id, card_data)
        hf_org = (
            hf_id.replace("datasets/", "").split("/", 1)[0].lower()
            if "/" in hf_id
            else ""
        )

        candidates = []
        seen = set()

        for match in cls.GITHUB_PATTERN.finditer(readme):
            root_match = cls.ROOT_PATTERN.match(match.group(0))
            if not root_match:
                continue

            repo_url = root_match.group(1)
            if repo_url in seen:
                continue
            seen.add(repo_url)

            owner, repo_name = repo_url.rsplit("/", 2)[-2:]
            if repo_name.lower() in cls.GENERIC_NAMES:
                continue

            repo_tokens = cls._tokenize(repo_name)
            similarity = max(
                (
                    cls._jaccard_similarity(cls._tokenize(alias), repo_tokens)
                    for alias in aliases
                ),
                default=0.0,
            )
            similarity += cls._version_bonus(hf_id, repo_name)

            if hf_org and (hf_org in owner.lower() or owner.lower() in hf_org):
                similarity += 0.05

            candidates.append((similarity, repo_url, owner))

        candidates.sort(reverse=True)

        high_confidence = [
            url for score, url, _ in candidates if score >= 0.40
        ]
        if high_confidence:
            return high_confidence

        same_owner = [
            url
            for _, url, owner in candidates
            if hf_org and hf_org in owner.lower()
        ]
        if same_owner:
            return [same_owner[0]]

        return [candidates[0][1]] if candidates else []


# ----------------- HF Client -----------------


class HFClient:
    """HuggingFace API client with GitHub URL extraction."""

    def __init__(self) -> None:
        self.token = (
            os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HF_TOKEN")
        )
        self.api = HfApi(token=self.token)
        self._session = _create_session(self.token)

    # ----- Model & Dataset info -----

    def get_model_info(self, hf_id: str) -> HFModelInfo:
        info = _retry(lambda: self.api.model_info(hf_id))
        return self._to_info(hf_id, info, repo_type="model")

    def get_dataset_info(self, hf_id: str) -> HFModelInfo:
        # DATASET: ensure we use dataset_info and normalize fields the same way
        info = _retry(lambda: self.api.dataset_info(hf_id))
        return self._to_info(hf_id, info, repo_type="dataset")

    # ----- Files -----

    def list_files(
        self, hf_id: str, *, repo_type: str = "model"
    ) -> List[HFFileInfo]:
        """
        Works for both models and datasets; pass repo_type="dataset"
        """
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

    # Convenience wrappers (optional but handy)
    def list_dataset_files(self, hf_id: str) -> List[HFFileInfo]:
        return self.list_files(hf_id, repo_type="dataset")

    def list_model_files(self, hf_id: str) -> List[HFFileInfo]:
        return self.list_files(hf_id, repo_type="model")

    # ----- README -----

    def get_readme(
        self,
        hf_id: str,
        *,
        revision: str = "main",
        repo_type: Optional[str] = None,
    ) -> str | None:
        """
        Fetch README with awareness of repo_type.
        For datasets, prefer datasets/<id>; for models, prefer <id>.
        """
        plain_id = hf_id.removeprefix("datasets/")
        dataset_id = f"datasets/{plain_id}"

        base_ids = (
            [dataset_id, plain_id]
            if repo_type == "dataset"
            else (
                [plain_id, dataset_id]
                if repo_type == "model"
                else [plain_id, dataset_id]
            )
        )

        for base_id in base_ids:
            for filename in ["README.md", "README.MD", "readme.md"]:
                url = (
                    f"https://huggingface.co/{base_id}/raw/{revision}/"
                    f"{filename}"
                )
                content = _retry(lambda: self._get_text(url))
                if content:
                    return content
        return None

    def get_dataset_readme(
        self, hf_id: str, *, revision: str = "main"
    ) -> str | None:
        # DATASET: explicit convenience for callers
        return self.get_readme(hf_id, revision=revision, repo_type="dataset")

    def get_model_readme(
        self, hf_id: str, *, revision: str = "main"
    ) -> str | None:
        return self.get_readme(hf_id, revision=revision, repo_type="model")

    # ----- model_index.json (rarely used for datasets) -----

    def get_model_index_json(
        self, hf_id: str, *, revision: str = "main"
    ) -> Dict | None:
        plain_id = hf_id.removeprefix("datasets/")
        urls = [
            f"https://huggingface.co/{plain_id}/raw/{revision}/"
            f"model_index.json",
            f"https://huggingface.co/datasets/{plain_id}/raw/{revision}/"
            f"model_index.json",
        ]
        for url in urls:
            data = _retry(lambda: self._get_json(url))
            if data is not None:
                return data
        return None

    # ----- GitHub URL extraction -----

    def get_github_urls(
        self,
        hf_id: str,
        readme: Optional[str] = None,
        card_data: Optional[Dict] = None,
    ) -> List[str]:
        """
        Extract GitHub URLs from README, fetching content if needed.
        Tries model README first, then dataset README (so it works for both).
        """
        if readme is None:
            readme = self.get_readme(
                hf_id, repo_type="model"
            ) or self.get_readme(hf_id, repo_type="dataset")
        if not readme:
            return []

        if card_data is None:
            # Try model -> dataset for card_data as well
            info = None
            try:
                info = self.get_model_info(hf_id)
            except Exception:
                try:
                    info = self.get_dataset_info(hf_id)
                except Exception:
                    pass
            card_data = info.card_data if info else None

        return GitHubMatcher.extract_urls(hf_id, readme, card_data)

    # ----- internal helpers -----

    def _to_info(
        self, hf_id: str, info: Any, *, repo_type: str
    ) -> HFModelInfo:
        """
        Normalize both model_info and dataset_info payloads into HFModelInfo.
        """
        card_data = _normalize_card_data(getattr(info, "cardData", None))

        # tags may be list[str] or None across versions
        raw_tags = getattr(info, "tags", None)
        if isinstance(raw_tags, (list, set, tuple)):
            tags = [str(t) for t in raw_tags]
        elif raw_tags is not None:
            tags = [str(raw_tags)]
        else:
            tags = []

        # Some hub versions expose downloads over last 30d as `downloads`;
        # all-time may be on `downloadsAllTime` (often None for datasets).
        downloads_30d = getattr(info, "downloads", None)
        downloads_all_time = getattr(info, "downloadsAllTime", None)

        created_at = getattr(info, "created_at", None)
        last_modified = getattr(info, "last_modified", None)
        gated = getattr(info, "gated", None)
        private = getattr(info, "private", None)
        likes = getattr(info, "likes", None)

        return HFModelInfo(
            hf_id=hf_id,
            card_data=card_data,
            tags=tags,
            likes=likes if isinstance(likes, int) else None,
            downloads_30d=(
                downloads_30d if isinstance(downloads_30d, int) else None
            ),
            downloads_all_time=(
                downloads_all_time
                if isinstance(downloads_all_time, int)
                else None
            ),
            created_at=str(created_at) if created_at is not None else None,
            last_modified=(
                str(last_modified) if last_modified is not None else None
            ),
            gated=bool(gated) if gated is not None else None,
            private=bool(private) if private is not None else None,
        )

    def _get_text(self, url: str) -> str | None:
        r = self._session.get(url)
        if r.status_code == 404:
            return None
        if r.status_code == 401:
            raise GatedRepoError(f"401 Unauthorized for {url}")
        r.raise_for_status()
        return r.text

    def _get_json(self, url: str) -> Dict | None:
        r = self._session.get(url)
        if r.status_code == 404:
            return None
        if r.status_code == 401:
            raise GatedRepoError(f"401 Unauthorized for {url}")
        r.raise_for_status()
        return r.json()


# Backward compatibility
def github_urls_from_readme(
    hf_id: str, readme: str, *, card_data: Dict | None = None
) -> List[str]:
    return GitHubMatcher.extract_urls(hf_id, readme, card_data)
