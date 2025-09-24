import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from collections.abc import Mapping

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class _TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, *args, timeout: int = 30, **kwargs):
        self._timeout = timeout
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        kwargs.setdefault("timeout", self._timeout)
        return super().send(request, **kwargs)


def _session() -> requests.Session:
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
    s.headers.update({
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "acme-model-evaluator/0.1 (+requests)",
    })
    return s


def _retry(fn, attempts: int = 4):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(0.4 * (2 ** i))


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
    VER = re.compile(
        r"(?:^|[^a-z0-9])v?(\d+(?:\.\d+)*)(?:[^a-z0-9]|$)", re.I
    )
    GENERIC = {
        "transformers", "diffusers", "datasets", "examples", "tutorials",
        "notebooks", "benchmarks", "models", "scripts", "awesome", "research",
    }

    @staticmethod
    def _norm(s: str) -> str:
        s = (s or "").lower()
        for pre in ("hf-", "huggingface-", "the-"):
            if s.startswith(pre):
                s = s[len(pre):]
        for suf in (
            "-dev", "-devkit", "-main", "-release", "-project", "-repo",
            "-code"
        ):
            if s.endswith(suf):
                s = s[: -len(suf)]
        return re.sub(r"[^a-z0-9]+", "-", s).strip("-")

    @staticmethod
    def _tokens(s: str) -> Set[str]:
        return {
            t for t in re.split(r"[^a-z0-9]+", GitHubMatcher._norm(s))
            if len(t) >= 3
        }

    @staticmethod
    def _aliases(hf_id: str, card: Dict | None) -> Set[str]:
        out: Set[str] = set()
        clean = hf_id.replace("datasets/", "") if hf_id else ""
        if "/" in clean:
            org, name = clean.split("/", 1)
            out.update({GitHubMatcher._norm(name), GitHubMatcher._norm(org)})
        else:
            out.add(GitHubMatcher._norm(clean))
        if card:
            mi = card.get("model-index")
            if isinstance(mi, list) and mi and isinstance(mi[0], dict):
                nm = mi[0].get("name")
                if isinstance(nm, str):
                    out.add(GitHubMatcher._norm(nm))
            for k in ("model_name", "name", "title"):
                v = card.get(k)
                if isinstance(v, str):
                    out.add(GitHubMatcher._norm(v))
        return {a for a in out if a}

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
        aliases = cls._aliases(hf_id, card)
        hf_org = (
            hf_id.replace("datasets/", "").split("/", 1)[0].lower()
            if "/" in hf_id else ""
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
                (cls._jac(cls._tokens(a), toks) for a in aliases),
                default=0.0,
            )
            sim += cls._ver_bonus(hf_id, repo)
            if hf_org and (
                hf_org in owner.lower() or owner.lower() in hf_org
            ):
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


# ---------------- HF client (public endpoints) ----------------

class HFClient:
    """
    Hugging Face client using only public endpoints.
    - /api/models/{id}, /api/datasets/{id}
    - 'siblings' for file listing
    - README and model_index.json via raw URLs
    """

    def __init__(self) -> None:
        self._http = _session()

    def _api_json(self, path: str) -> Dict[str, Any] | None:
        url = f"https://huggingface.co{path}"
        r = _retry(lambda: self._http.get(url))
        if r.status_code in (401, 404):
            return None
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return None

    # ---- Info ----

    def get_model_info(self, hf_id: str) -> HFModelInfo:
        data = self._api_json(f"/api/models/{hf_id}")
        if not data:
            raise FileNotFoundError(f"Model not found: {hf_id}")
        return self._to_info(hf_id, data)

    def get_dataset_info(self, hf_id: str) -> HFModelInfo:
        data = self._api_json(f"/api/datasets/{hf_id}")
        if not data:
            raise FileNotFoundError(f"Dataset not found: {hf_id}")
        return self._to_info(hf_id, data)

    # ---- Files (siblings) ----

    def list_files(
        self, hf_id: str, *, repo_type: str = "model"
    ) -> List[HFFileInfo]:
        path = (
            "/api/models/" + hf_id if repo_type == "model"
            else "/api/datasets/" + hf_id
        )
        data = self._api_json(path) or {}
        files: List[HFFileInfo] = []
        sibs = data.get("siblings", [])
        if isinstance(sibs, list):
            for s in sibs:
                rfn = s.get("rfilename")
                if isinstance(rfn, str) and rfn:
                    files.append(HFFileInfo(path=rfn, size=s.get("size")))
        return files

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
    ) -> str | None:
        plain = hf_id.removeprefix("datasets/")
        ds_id = f"datasets/{plain}"
        order = (
            [plain, ds_id] if repo_type == "model"
            else [ds_id, plain] if repo_type == "dataset"
            else [plain, ds_id]
        )
        for base in order:
            for fn in ("README.md", "README.MD", "readme.md"):
                url = f"https://huggingface.co/{base}/raw/{revision}/{fn}"
                r = _retry(lambda: self._http.get(url))
                if r.status_code == 200 and r.text:
                    return r.text
        return None

    def get_dataset_readme(
        self, hf_id: str, *, revision: str = "main"
    ) -> str | None:
        return self.get_readme(hf_id, revision=revision, repo_type="dataset")

    def get_model_readme(
        self, hf_id: str, *, revision: str = "main"
    ) -> str | None:
        return self.get_readme(hf_id, revision=revision, repo_type="model")

    # ---- model_index.json (raw) ----

    def get_model_index_json(
        self, hf_id: str, *, revision: str = "main"
    ) -> Dict | None:
        plain = hf_id.removeprefix("datasets/")
        urls = [
            (f"https://huggingface.co/{plain}/raw/{revision}/"
             "model_index.json"),
            (f"https://huggingface.co/datasets/{plain}/raw/{revision}/"
             "model_index.json"),
        ]
        for url in urls:
            r = _retry(lambda: self._http.get(url))
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
            readme = (
                self.get_readme(hf_id, repo_type="model")
                or self.get_readme(hf_id, repo_type="dataset")
            )
        if not readme:
            return []
        if card_data is None:
            data = (
                self._api_json(f"/api/models/{hf_id}")
                or self._api_json(f"/api/datasets/{hf_id}")
            )
            if data and isinstance(data.get("cardData"), dict):
                card_data = data["cardData"]
        return GitHubMatcher.extract_urls(hf_id, readme, card_data)

    # ---- Normalizer ----

    def _to_info(self, hf_id: str, data: Dict[str, Any]) -> HFModelInfo:
        card = _normalize_card_data(data.get("cardData"))
        tags = (
            [str(t) for t in data.get("tags", [])]
            if isinstance(data.get("tags"), list) else []
        )
        likes = (
            data.get("likes") if isinstance(data.get("likes"), int) else None
        )
        d30 = (
            data.get("downloads")
            if isinstance(data.get("downloads"), int) else None
        )
        dall = (
            data.get("downloadsAllTime")
            if isinstance(data.get("downloadsAllTime"), int) else None
        )
        created = (
            str(data.get("createdAt"))
            if data.get("createdAt") is not None else None
        )
        modified = (
            str(data.get("lastModified"))
            if data.get("lastModified") is not None else None
        )
        gated = (
            bool(data.get("gated"))
            if data.get("gated") is not None else None
        )
        private = (
            bool(data.get("private"))
            if data.get("private") is not None else None
        )
        return HFModelInfo(
            hf_id=hf_id,
            card_data=card,
            tags=tags,
            likes=likes,
            downloads_30d=d30,
            downloads_all_time=dall,
            created_at=created,
            last_modified=modified,
            gated=gated,
            private=private,
        )


# Back-compat shim
def github_urls_from_readme(
    hf_id: str, readme: str, *, card_data: Dict | None = None
) -> List[str]:
    return GitHubMatcher.extract_urls(hf_id, readme, card_data)
