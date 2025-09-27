import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from collections.abc import Mapping

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from huggingface_hub import HfApi  # type: ignore
except Exception:  # pragma: no cover
    HfApi = None  # type: ignore


def _removeprefix(s: str, prefix: str) -> str:
    return s[len(prefix):] if s.startswith(prefix) else s


_WEIGHT_EXTS = {
    "safetensors", "bin", "pt", "pth", "h5", "onnx", "tflite", "gguf", "ggml", "ckpt"
}


def _ext_of(path: str) -> str:
    p = path.rsplit("/", 1)[-1]
    if "." in p:
        return p.rsplit(".", 1)[-1].lower()
    return ""
# ---------------- HTTP plumbing ----------------


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
            time.sleep(0.4 * (2 ** i))


def _json_get(session: requests.Session, url: str) -> Optional[Dict[str, Any]]:
    r = _retry(lambda: session.get(url))
    if r.status_code in (404, 401):
        return None
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return None


def _text_get(session: requests.Session, url: str) -> Optional[str]:
    r = _retry(lambda: session.get(url))
    if r.status_code != 200:
        return None
    return r.text or None


# ---------------- Data models ----------------

@dataclass
class HFModelInfo:
    hf_id: str
    card_data: Optional[Dict[str, Any]]
    tags: List[str]
    likes: Optional[int]
    downloads_30d: Optional[int]
    downloads_all_time: Optional[int]
    created_at: Optional[str]
    last_modified: Optional[str]
    gated: Optional[bool]
    private: Optional[bool]


@dataclass
class HFFileInfo:
    path: str
    size: Optional[int]


# ---------------- Helpers ----------------

def _normalize_card_data(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    try:
        if hasattr(x, "to_dict") and callable(x.to_dict):
            return dict(x.to_dict())
        if hasattr(x, "data") and isinstance(x.data, Mapping):
            return dict(x.data)
        if isinstance(x, Mapping):
            return dict(x)
        for m in ("model_dump", "dict", "json"):
            if hasattr(x, m):
                v = getattr(x, m)()
                return json.loads(v) if m == "json" else dict(v)
    except Exception:
        pass
    return {}


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


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
        for suf in ("-dev", "-devkit", "-main", "-release", "-project", "-repo", "-code"):
            if s.endswith(suf):
                s = s[: -len(suf)]
        return re.sub(r"[^a-z0-9]+", "-", s).strip("-")

    @staticmethod
    def _normalize(s: str) -> str:
        return GitHubMatcher._norm(s)

    @staticmethod
    def _tokens(s: str) -> Set[str]:
        return {t for t in re.split(r"[^a-z0-9]+", GitHubMatcher._norm(s)) if len(t) >= 3}

    @staticmethod
    def _jac(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        return inter / len(a | b) if inter else 0.0

    @staticmethod
    def _ver_bonus(hf_id: str, repo: str) -> float:
        hv = [tuple(
            map(int, m.group(1).split("."))) for m in GitHubMatcher.VER.finditer(hf_id or "")]
        rv = [tuple(
            map(int, m.group(1).split("."))) for m in GitHubMatcher.VER.finditer(repo or "")]
        if not hv or not rv:
            return 0.0
        return max(1.0 if x == y else 0.9 for x in hv for y in rv)

    @classmethod
    def extract_urls(
            cls, hf_id: str, readme: str, card: Optional[Dict[str, Any]] = None) -> List[str]:
        if not readme:
            return []
        aliases: Set[str] = set([cls._normalize(hf_id)])
        if card and isinstance(card.get("name"), str):
            aliases = aliases.union({cls._normalize(card.get("name"))})  # type: ignore[arg-type]

        hf_org = hf_id.replace("datasets/", "").split("/", 1)[0].lower() if "/" in hf_id else ""
        cands: List[Tuple[float, str, str]] = []
        seen: Set[str] = set()

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
            sim = 0.0
            for a in aliases:
                sim = max(sim, cls._jac(cls._tokens(a), toks))
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


# ---------------- HF client (public endpoints first) ----------------

class HFClient:
    """
    Hugging Face client that uses public endpoints (no auth) for correctness,
    with optional HfApi fallback when available.
    """

    def __init__(self) -> None:
        self.api = HfApi() if HfApi else None  # optional
        self._session = _create_session()

    # ---- Public API helpers ----

    def _api_model_json(self, hf_id: str) -> Optional[Dict[str, Any]]:
        return _json_get(self._session, f"https://huggingface.co/api/models/{hf_id}")

    def _api_dataset_json(self, hf_id: str) -> Optional[Dict[str, Any]]:
        return _json_get(self._session, f"https://huggingface.co/api/datasets/{hf_id}")

    def _api_tree_json(self, base: str, revision: str = "main") -> Optional[List[Dict[str, Any]]]:
        url = f"https://huggingface.co/api/{base}/tree/{revision}?recursive=1"
        r = _retry(lambda: self._session.get(url))
        if r.status_code != 200:
            return None
        try:
            data = r.json()
            return data if isinstance(data, list) else None
        except Exception:
            return None

    # ---- File listing (robust) ----

    def _extract_files_from_api_obj(self, obj: Dict[str, Any]) -> List[HFFileInfo]:
        out: List[HFFileInfo] = []
        sibs = obj.get("siblings")
        if isinstance(sibs, list):
            for s in sibs:
                p = s.get("rfilename") if isinstance(s, dict) else None
                if isinstance(p, str) and p:
                    out.append(HFFileInfo(path=p, size=_safe_int(s.get("size"))))
        return out

    def list_files(self, hf_id: str, *, repo_type: str = "model") -> List[HFFileInfo]:
        # 1) siblings
        if repo_type == "dataset":
            obj = self._api_dataset_json(hf_id)
            base_for_tree = f"datasets/{hf_id}"
            html_base = f"datasets/{hf_id}"
        else:
            obj = self._api_model_json(hf_id)
            base_for_tree = f"models/{hf_id}"
            html_base = hf_id

        files = self._extract_files_from_api_obj(obj or {})
        if files:
            return self._fill_missing_sizes(hf_id, files, repo_type=repo_type, revision="main")

        # 2) tree
        tree = self._api_tree_json(base_for_tree, revision="main")
        if tree:
            out: List[HFFileInfo] = []
            for node in tree:
                if node.get("type") == "file" and isinstance(node.get("path"), str):
                    out.append(HFFileInfo(path=node["path"], size=_safe_int(node.get("size"))))
            if out:
                return self._fill_missing_sizes(hf_id, out, repo_type=repo_type, revision="main")

        # 3) huggingface_hub
        out: List[HFFileInfo] = []
        if self.api:
            try:
                names = self.api.list_repo_files(hf_id, repo_type=repo_type)
                sizes: Dict[str, Optional[int]] = {}
                try:
                    infos = self.api.get_paths_info(hf_id, paths=names, repo_type=repo_type)
                    for info in infos or []:
                        p = getattr(info, "path", None)
                        sz = getattr(info, "size", None)
                        if isinstance(p, str):
                            sizes[p] = sz if isinstance(sz, int) else None
                except Exception:
                    sizes = {}
                out = [HFFileInfo(path=f, size=sizes.get(f)) for f in names]
                if out:
                    return self._fill_missing_sizes(
                        hf_id, out, repo_type=repo_type, revision="main")
            except Exception:
                pass

        # 4) HTML
        return self._parse_tree_html(html_base, revision="main")

    def _parse_tree_html(
            self, base_path: str, revision: str = "main") -> List[HFFileInfo]:
        url = f"https://huggingface.co/{base_path}/tree/{revision}"
        html = _text_get(self._session, url)
        if not html:
            return []
        base_quoted = re.escape(base_path)
        rev_quoted = re.escape(revision)
        pat = re.compile(rf"/{base_quoted}/blob/{rev_quoted}/([^\"'#?]+)")
        files = {m.group(1) for m in pat.finditer(html)}
        return [HFFileInfo(path=f, size=None) for f in sorted(files)]

    def list_model_files(self, hf_id: str, *, revision: str = "main") -> List[HFFileInfo]:
        obj = self._api_model_json(hf_id)
        files = self._extract_files_from_api_obj(obj or {})
        if files:
            return self._fill_missing_sizes(hf_id, files, repo_type="model", revision=revision)

        tree = self._api_tree_json(f"models/{hf_id}", revision=revision)
        if tree:
            out: List[HFFileInfo] = []
            for node in tree:
                if node.get("type") == "file" and isinstance(node.get("path"), str):
                    out.append(HFFileInfo(path=node["path"], size=_safe_int(node.get("size"))))
            if out:
                return self._fill_missing_sizes(hf_id, out, repo_type="model", revision=revision)

        if self.api:
            try:
                names = self.api.list_repo_files(hf_id, repo_type="model")
                sizes: Dict[str, Optional[int]] = {}
                try:
                    infos = self.api.get_paths_info(hf_id, paths=names, repo_type="model")
                    for info in infos or []:
                        p = getattr(info, "path", None)
                        sz = getattr(info, "size", None)
                        if isinstance(p, str):
                            sizes[p] = sz if isinstance(sz, int) else None
                except Exception:
                    sizes = {}
                out = [HFFileInfo(path=f, size=sizes.get(f)) for f in names]
                if out:
                    return self._fill_missing_sizes(
                        hf_id, out, repo_type="model", revision=revision)
            except Exception:
                pass

        return self._parse_tree_html(hf_id, revision=revision)
    
    def _resolve_url(
            self, hf_id: str, path: str, *,
            revision: str = "main", repo_type: str = "model") -> str:
        base = f"datasets/{hf_id}" if repo_type == "dataset" else hf_id
        return f"https://huggingface.co/{base}/resolve/{revision}/{path}"

    # HEAD request to get content-length
    def _head_size(self, url: str) -> Optional[int]:
        try:
            r = _retry(lambda: self._session.head(url, allow_redirects=True))
            if r.status_code == 200:
                cl = r.headers.get("Content-Length")
                if cl is not None:
                    try:
                        return int(cl)
                    except ValueError:
                        return None
        except Exception:
            return None
        return None

    # Fill in missing sizes for likely-weight files
    def _fill_missing_sizes(
        self,
        hf_id: str,
        files: List["HFFileInfo"],
        *,
        repo_type: str = "model",
        revision: str = "main",
        only_weight_exts: bool = True,
        max_heads: int = 40,
    ) -> List["HFFileInfo"]:
        if not files:
            return files

        filled = 0
        out: List[HFFileInfo] = []
        for f in files:
            if f.size is not None:
                out.append(f)
                continue

            ext = _ext_of(f.path)
            if only_weight_exts and ext not in _WEIGHT_EXTS:
                out.append(f)  # leave as None
                continue

            if filled >= max_heads:
                out.append(f)
                continue

            url = self._resolve_url(hf_id, f.path, revision=revision, repo_type=repo_type)
            sz = self._head_size(url)
            if sz is not None:
                out.append(HFFileInfo(path=f.path, size=sz))
                filled += 1
            else:
                out.append(f)
        return out
    
    def list_dataset_files(self, hf_id: str, *, revision: str = "main") -> List[HFFileInfo]:
        obj = self._api_dataset_json(hf_id)
        files = self._extract_files_from_api_obj(obj or {})
        if files:
            return self._fill_missing_sizes(hf_id, files, repo_type="dataset", revision=revision)

        tree = self._api_tree_json(f"datasets/{hf_id}", revision=revision)
        if tree:
            out2: List[HFFileInfo] = []
            for node in tree:
                if node.get("type") == "file" and isinstance(node.get("path"), str):
                    out2.append(HFFileInfo(path=node["path"], size=_safe_int(node.get("size"))))
            if out2:
                return self._fill_missing_sizes(hf_id, out2, repo_type="dataset", revision=revision)

        if self.api:
            try:
                names = self.api.list_repo_files(hf_id, repo_type="dataset")
                sizes: Dict[str, Optional[int]] = {}
                try:
                    infos = self.api.get_paths_info(hf_id, paths=names, repo_type="dataset")
                    for info in infos or []:
                        p = getattr(info, "path", None)
                        sz = getattr(info, "size", None)
                        if isinstance(p, str):
                            sizes[p] = sz if isinstance(sz, int) else None
                except Exception:
                    sizes = {}
                out = [HFFileInfo(path=f, size=sizes.get(f)) for f in names]
                if out:
                    return self._fill_missing_sizes(
                        hf_id, out, repo_type="dataset", revision=revision)
            except Exception:
                pass

        return self._parse_tree_html(f"datasets/{hf_id}", revision=revision)

    # ---- README (raw) ----

    def get_readme(
        self,
        hf_id: str,
        *,
        revision: str = "main",
        repo_type: Optional[str] = None,
    ) -> Optional[str]:
        plain = _removeprefix(hf_id, "datasets/")
        ds_id = "datasets/" + plain
        if repo_type == "model":
            order: List[str] = [plain, ds_id]
        elif repo_type == "dataset":
            order = [ds_id, plain]
        else:
            order = [plain, ds_id]
        for base in order:
            for fn in ("README.md", "README.MD", "readme.md"):
                url = f"https://huggingface.co/{base}/raw/{revision}/{fn}"
                txt = _text_get(self._session, url)
                if txt:
                    return txt
        return None

    def get_dataset_readme(self, hf_id: str, *, revision: str = "main") -> Optional[str]:
        return self.get_readme(hf_id, revision=revision, repo_type="dataset")

    def get_model_readme(self, hf_id: str, *, revision: str = "main") -> Optional[str]:
        return self.get_readme(hf_id, revision=revision, repo_type="model")

    # ---- model_index.json (raw) ----

    def get_model_index_json(
            self, hf_id: str, *, revision: str = "main") -> Optional[Dict[str, Any]]:
        plain = _removeprefix(hf_id, "datasets/")
        urls = [
            f"https://huggingface.co/{plain}/raw/{revision}/model_index.json",
            f"https://huggingface.co/datasets/{plain}/raw/{revision}/model_index.json",
        ]
        for url in urls:
            obj = _json_get(self._session, url)
            if obj:
                return obj
        return None

    # ---- Card/info (public API first) ----

    def _to_info_from_api_obj(self, hf_id: str, obj: Dict[str, Any]) -> HFModelInfo:
        card = _normalize_card_data(obj.get("cardData"))
        tags_raw = obj.get("tags") or []
        tags = [str(t) for t in tags_raw] if isinstance(tags_raw, list) else []
        likes = _safe_int(obj.get("likes"))
        d30 = _safe_int(obj.get("downloads"))
        dall = _safe_int(obj.get("downloadsAllTime"))
        created = obj.get("createdAt")
        modified = obj.get("lastModified")
        gated = bool(obj.get("gated")) if obj.get("gated") is not None else None
        private = bool(obj.get("private")) if obj.get("private") is not None else None
        rid = obj.get("modelId") or obj.get("datasetId") or hf_id
        return HFModelInfo(
            hf_id=str(rid),
            card_data=card,
            tags=tags,
            likes=likes,
            downloads_30d=d30,
            downloads_all_time=dall,
            created_at=str(created) if created is not None else None,
            last_modified=str(modified) if modified is not None else None,
            gated=gated,
            private=private,
        )

    def get_model_info(self, hf_id: str) -> HFModelInfo:
        obj = self._api_model_json(hf_id)
        if obj:
            return self._to_info_from_api_obj(hf_id, obj)
        if self.api:
            try:
                hub_obj = self.api.model_info(hf_id)
                return self._to_info_from_api_obj(
                    hf_id,
                    {
                        "cardData": getattr(hub_obj, "cardData", None),
                        "tags": getattr(hub_obj, "tags", None),
                        "likes": getattr(hub_obj, "likes", None),
                        "downloads": getattr(hub_obj, "downloads", None),
                        "downloadsAllTime": getattr(hub_obj, "downloadsAllTime", None),
                        "createdAt": getattr(hub_obj, "createdAt", None),
                        "lastModified": getattr(hub_obj, "lastModified", None),
                        "gated": getattr(hub_obj, "gated", None),
                        "private": getattr(hub_obj, "private", None),
                        "modelId": getattr(hub_obj, "modelId", hf_id),
                    },
                )
            except Exception:
                pass
        return HFModelInfo(
            hf_id=hf_id,
            card_data={},
            tags=[],
            likes=None,
            downloads_30d=None,
            downloads_all_time=None,
            created_at=None,
            last_modified=None,
            gated=None,
            private=None,
        )

    def get_dataset_info(self, hf_id: str) -> HFModelInfo:
        obj = self._api_dataset_json(hf_id)
        if obj:
            return self._to_info_from_api_obj(hf_id, obj)
        if self.api:
            try:
                hub_obj = self.api.dataset_info(hf_id)
                return self._to_info_from_api_obj(
                    hf_id,
                    {
                        "cardData": getattr(hub_obj, "cardData", None),
                        "tags": getattr(hub_obj, "tags", None),
                        "likes": getattr(hub_obj, "likes", None),
                        "downloads": getattr(hub_obj, "downloads", None),
                        "downloadsAllTime": getattr(hub_obj, "downloadsAllTime", None),
                        "createdAt": getattr(hub_obj, "createdAt", None),
                        "lastModified": getattr(hub_obj, "lastModified", None),
                        "gated": getattr(hub_obj, "gated", None),
                        "private": getattr(hub_obj, "private", None),
                        "datasetId": getattr(hub_obj, "datasetId", hf_id),
                    },
                )
            except Exception:
                pass
        return HFModelInfo(
            hf_id=hf_id,
            card_data={},
            tags=[],
            likes=None,
            downloads_30d=None,
            downloads_all_time=None,
            created_at=None,
            last_modified=None,
            gated=None,
            private=None,
        )

    # ---- GitHub URLs from README ----

    def get_github_urls(
        self,
        hf_id: str,
        readme: Optional[str] = None,
        card_data: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if readme is None:
            readme = self.get_readme(
                hf_id, repo_type="model") or self.get_readme(hf_id, repo_type="dataset")
        if not readme:
            return []
        if card_data is None:
            data = self._api_model_json(hf_id) or self._api_dataset_json(hf_id)
            if data and isinstance(data.get("cardData"), dict):
                card_data = data["cardData"]
        return GitHubMatcher.extract_urls(hf_id, readme, card_data)
