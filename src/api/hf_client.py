import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from collections.abc import Mapping

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from huggingface_hub import HfApi


class _TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, *args, timeout: int = 30, **kwargs):
        self._timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        kwargs.setdefault("timeout", self._timeout)
        return super().send(request, **kwargs)


def _create_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = _TimeoutHTTPAdapter(max_retries=retry, timeout=30)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(
        {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "acme-model-evaluator/0.1 (+requests)",
        }
    )
    return s


def _retry(fn, attempts: int = 4):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(0.4 * (2**i))


# ---------------- Data models ----------------


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


# ---------------- Helpers ----------------


def _normalize_card_data(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    try:
        if hasattr(x, "to_dict") and callable(x.to_dict):  # type: ignore
            return dict(x.to_dict())  # type: ignore[attr-defined]
        if hasattr(x, "data") and isinstance(x.data, Mapping):  # type: ignore
            return dict(x.data)  # type: ignore[attr-defined]
        if isinstance(x, Mapping):
            return dict(x)
        for m in ("model_dump", "dict", "json"):
            if hasattr(x, m):
                v = getattr(x, m)()
                return json.loads(v) if m == "json" else dict(v)
    except Exception:
        pass
    return {}


# ---------------- GitHub URL extraction ----------------


class GitHubMatcher:
    GITHUB = re.compile(r"https?://github\.com/[^\s)\"'<>]+", re.I)
    ROOT = re.compile(r"^(https?://github\.com/[-\w.]+/[-\w.]+)", re.I)
    VER = re.compile(r"(?:^|[^a-z0-9])v?(\d+(?:\.\d+)*)(?:[^a-z0-9]|$)", re.I)
    GENERIC = {
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
    def _norm(s: str) -> str:
        s = (s or "").lower()
        for pre in ("hf-", "huggingface-", "the-"):
            if s.startswith(pre):
                s = s[len(pre):]
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
        return re.sub(r"[^a-z0-9]+", "-", s).strip("-")

    # These two aliases satisfy tests that call _normalize/_tokenize
    @staticmethod
    def _normalize(s: str) -> str:
        return GitHubMatcher._norm(s)

    @staticmethod
    def _tokens(s: str) -> Set[str]:
        return {
            t
            for t in re.split(r"[^a-z0-9]+", GitHubMatcher._norm(s))
            if len(t) >= 3
        }

    @staticmethod
    def _tokenize(s: str) -> Set[str]:
        return GitHubMatcher._tokens(s)

    @staticmethod
    def _jac(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        return inter / len(a | b) if inter else 0.0

    @staticmethod
    def _ver_bonus(hf_id: str, repo: str) -> float:
        hv = [
            tuple(map(int, m.group(1).split(".")))
            for m in GitHubMatcher.VER.finditer(hf_id or "")
        ]
        rv = [
            tuple(map(int, m.group(1).split(".")))
            for m in GitHubMatcher.VER.finditer(repo or "")
        ]
        if not hv or not rv:
            return 0.0
        return max(1.0 if x == y else 0.9 for x in hv for y in rv)

    @classmethod
    def extract_urls(
        cls, hf_id: str, readme: str, card: Dict | None = None
    ) -> List[str]:
        if not readme:
            return []
        aliases = {cls._normalize(hf_id)} | (
            {cls._normalize(card.get("name"))}
            if card and isinstance(card.get("name"), str)
            else set()
        )
        hf_org = (
            hf_id.replace("datasets/", "").split("/", 1)[0].lower()
            if "/" in hf_id
            else ""
        )
        cands: List[tuple[float, str, str]] = []
        seen: set[str] = set()

        for m in cls.GITHUB.finditer(readme):
            root = cls.ROOT.match(m.group(0))
            if not root:
                continue
            repo_url = root.group(1)
            if repo_url in seen:
                continue
            seen.add(repo_url)
            owner, repo = repo_url.rsplit("/", 2)[-2:]
            if repo.lower() in cls.GENERIC:
                continue
            toks = cls._tokens(repo)
            sim = max(
                (cls._jac(cls._tokens(a), toks) for a in aliases), default=0.0
            )
            sim += cls._ver_bonus(hf_id, repo)
            if hf_org and (hf_org in owner.lower() or owner.lower() in hf_org):
                sim += 0.05
            cands.append((sim, repo_url, owner))

        cands.sort(reverse=True)
        good = [u for s, u, _ in cands if s >= 0.40]
        if good:
            return good
        same_owner = [u for _, u, o in cands if hf_org and hf_org in o.lower()]
        if same_owner:
            return [same_owner[0]]
        return [cands[0][1]] if cands else []


# ---------------- HF client (public endpoints; no tokens) ----------------


class HFClient:
    """
    Hugging Face client using only public endpoints (no auth/token).
    - Uses HfApi for convenient listing when available (tests patch it).
    - Uses raw GETs for README/model_index.
    """

    def __init__(self) -> None:
        self.api = HfApi()  # available for tests to patch
        self._session = _create_session()  # public session, no token

    # ---- Low-level helpers ----

    def _api_json(self, path: str) -> Dict[str, Any] | None:
        url = f"https://huggingface.co{path}"
        r = _retry(lambda: self._session.get(url))
        if r.status_code in (401, 404):
            return None
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return None

    def _get_text(self, url: str) -> Optional[str]:
        r = _retry(lambda: self._session.get(url))
        if r.status_code == 200 and r.text:
            return r.text
        return None

    # ---- Info via HfApi objects (tests patch these) ----

    def _to_info_from_hub(self, hf_id: str, hub_obj: Any) -> HFModelInfo:
        # normalize diverse attribute shapes
        card = _normalize_card_data(getattr(hub_obj, "cardData", None))
        tags = [str(t) for t in (getattr(hub_obj, "tags", None) or [])]
        likes = getattr(hub_obj, "likes", None)
        d30 = getattr(hub_obj, "downloads", None)
        dall = getattr(hub_obj, "downloadsAllTime", None)
        created = getattr(hub_obj, "createdAt", None)
        modified = getattr(hub_obj, "lastModified", None)
        gated = getattr(hub_obj, "gated", None)
        private = getattr(hub_obj, "private", None)
        # some objects may call modelId/datasetId
        if not hf_id:
            hf_id = (
                getattr(hub_obj, "modelId", None)
                or getattr(hub_obj, "datasetId", None)
                or ""
            )
        return HFModelInfo(
            hf_id=hf_id,
            card_data=card,
            tags=tags,
            likes=likes if isinstance(likes, int) else None,
            downloads_30d=d30 if isinstance(d30, int) else None,
            downloads_all_time=dall if isinstance(dall, int) else None,
            created_at=str(created) if created is not None else None,
            last_modified=str(modified) if modified is not None else None,
            gated=bool(gated) if gated is not None else None,
            private=bool(private) if private is not None else None,
        )

    def get_model_info(self, hf_id: str) -> HFModelInfo:
        info = self.api.model_info(hf_id)  # tests patch this
        return self._to_info_from_hub(hf_id, info)

    def get_dataset_info(self, hf_id: str) -> HFModelInfo:
        info = self.api.dataset_info(hf_id)  # tests patch this
        return self._to_info_from_hub(hf_id, info)

    # ---- Files (prefer HfApi for tests; fallback sizes unknown) ----

    def list_files(
        self, hf_id: str, *, repo_type: str = "model"
    ) -> List[HFFileInfo]:
        try:
            files = self.api.list_repo_files(hf_id, repo_type=repo_type)
        except Exception:
            files = []
        out: List[HFFileInfo] = []
        if files:
            # try to enrich sizes if get_paths_info exists
            sizes: Dict[str, int] = {}
            try:
                infos = self.api.get_paths_info(
                    hf_id, paths=files, repo_type=repo_type
                )
                for info in infos or []:
                    p = getattr(info, "path", None)
                    sz = getattr(info, "size", None)
                    if isinstance(p, str):
                        sizes[p] = sz if isinstance(sz, int) else None
            except AttributeError:
                # tests expect to handle AttributeError and return size=None
                sizes = {}
            except Exception:
                sizes = {}
            for f in files:
                out.append(HFFileInfo(path=f, size=sizes.get(f)))
        return out

    def list_dataset_files(self, hf_id: str) -> List[HFFileInfo]:
        return self.list_files(hf_id, repo_type="dataset")

    def list_model_files(self, hf_id: str) -> List[HFFileInfo]:
        return self.list_files(hf_id, repo_type="model")

    # ---- README (raw) ----

    def get_readme(
        self,
        hf_id: str,
        *,
        revision: str = "main",
        repo_type: Optional[str] = None,
    ) -> Optional[str]:
        plain = hf_id.removeprefix("datasets/")
        ds_id = f"datasets/{plain}"
        order = (
            [plain, ds_id]
            if repo_type == "model"
            else [ds_id, plain] if repo_type == "dataset" else [plain, ds_id]
        )
        for base in order:
            for fn in ("README.md", "README.MD", "readme.md"):
                url = f"https://huggingface.co/{base}/raw/{revision}/{fn}"
                txt = self._get_text(url)
                if txt:
                    return txt
        return None

    def get_dataset_readme(
        self, hf_id: str, *, revision: str = "main"
    ) -> Optional[str]:
        return self.get_readme(hf_id, revision=revision, repo_type="dataset")

    def get_model_readme(
        self, hf_id: str, *, revision: str = "main"
    ) -> Optional[str]:
        return self.get_readme(hf_id, revision=revision, repo_type="model")

    # ---- model_index.json (raw) ----

    def get_model_index_json(
        self, hf_id: str, *, revision: str = "main"
    ) -> Optional[Dict]:
        plain = hf_id.removeprefix("datasets/")
        urls = [
            (
                f"https://huggingface.co/{plain}/raw/{revision}/model_index.json"
            ),
            (
                f"https://huggingface.co/datasets/{plain}/raw/{revision}/model_index.json"
            ),
        ]
        for url in urls:
            r = _retry(lambda: self._session.get(url))
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return None
        return None

    # ---- GitHub URLs from README ----

    def get_github_urls(
        self,
        hf_id: str,
        readme: Optional[str] = None,
        card_data: Optional[Dict] = None,
    ) -> List[str]:
        if readme is None:
            readme = self.get_readme(
                hf_id, repo_type="model"
            ) or self.get_readme(hf_id, repo_type="dataset")
        if not readme:
            return []
        if card_data is None:
            # Best-effort to grab card data (no token)
            data = self._api_json(f"/api/models/{hf_id}") or self._api_json(
                f"/api/datasets/{hf_id}"
            )
            if data and isinstance(data.get("cardData"), dict):
                card_data = data["cardData"]
        return GitHubMatcher.extract_urls(hf_id, readme, card_data)


# Back-compat shim
def github_urls_from_readme(
    hf_id: str, readme: str, *, card_data: Dict | None = None
) -> List[str]:
    return GitHubMatcher.extract_urls(hf_id, readme, card_data)
