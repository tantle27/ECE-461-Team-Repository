# metrics/ramp_up_time_metric.py

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from api.llm_client import LLMClient
from .base_metric import BaseMetric


# -------- tuning knobs --------
@dataclass(frozen=True)
class MixHeu:
    quickstart: float = 0.36
    examples: float = 0.30
    depth: float = 0.19
    ext: float = 0.15


@dataclass(frozen=True)
class MixLLM:
    base: float = 0.62
    max_conf_add: float = 0.28
    swing: float = 0.56


class RampUpTimeMetric(BaseMetric):
    def __init__(self, weight: float = 0.15, use_llm: bool = True):
        super().__init__(name="RampUpTime", weight=weight)
        self._use_llm = bool(use_llm)
        self._llm = LLMClient()
        self._mh = MixHeu()
        self._ml = MixLLM()

    # ---- entry point ----
    def evaluate(self, repo_context: dict) -> float:
        ctx = repo_context.get("_ctx_obj")

        h_score = self._readme_signal_score(repo_context, ctx)

        floor = self._context_floor(ctx)
        base = max(h_score, floor)

        if not (self._use_llm and getattr(self._llm, "provider", None)):
            return self._clip01(base)

        try:
            llm_score, conf, parts = self._llm_view(repo_context, ctx)
            repo_context["_rampup_llm_parts"] = parts

            conf_add = 0.6 * max(0.0, conf - 0.5)
            w = min(0.95, self._ml.base + min(self._ml.max_conf_add, conf_add))
            mix = w * llm_score + (1.0 - w) * base

            # bound swing relative to the heuristic+floor baseline
            delta = self._bound(mix - base, -self._ml.swing, self._ml.swing)
            out = base + delta

            if parts.get("any_signal"):
                out = max(out, 0.32, floor)

            return self._clip01(out)
        except Exception:
            return self._clip01(base)

    def get_description(self) -> str:
        return (
            "Ramp-up score using provenance/popularity floors, a combined README heuristic, "
            "and an optional LLM pass."
        )

    # ---- README composition + heuristic scorer ----
    def _compose_readme(self, repo_context: dict, ctx: Any) -> str:
        base_text = (repo_context.get("readme_text") or "").strip()
        best_code = ""
        if ctx is not None:
            for c in getattr(ctx, "linked_code", []) or []:
                t = (getattr(c, "readme_text", None) or "").strip()
                if len(t) > len(best_code):
                    best_code = t
        if best_code and best_code not in base_text:
            suffix = "\n\n---\n# Linked code\n" + best_code[:7000]
            return base_text + suffix
        return base_text

    def _readme_signal_score(self, repo_context: dict, ctx: Any) -> float:
        text = self._compose_readme(repo_context, ctx)
        r = text.lower()

        install = any(
            k in r
            for k in (
                "pip install",
                "pip3 install",
                "conda install",
                "poetry add",
                "requirements.txt",
            )
        )
        howto = any(
            k in r
            for k in ("getting started", "quickstart", "quick start", "setup", "how to")
        )
        usage = any(
            k in r
            for k in (
                "usage",
                "example",
                "inference",
                "train",
                "run",
                "predict",
                "fine-tune",
                "finetune",
            )
        )
        fenced = ("```" in r) or ("$ " in r)

        notebook = any(k in r for k in ("colab", ".ipynb", "notebook"))
        demo = any(k in r for k in ("demo", "sample", "samples", "examples"))

        api = any(k in r for k in ("api reference", "api docs", "reference", "documentation"))
        cfg = any(k in r for k in ("configuration", "config", "settings", "parameters", "options"))
        trbl = any(k in r for k in ("troubleshooting", "faq", "known issues"))
        docs_dir = ("docs/" in r) or ("mkdocs" in r) or ("sphinx" in r)

        ext = any(
            k in r
            for k in ("readthedocs", "https://docs.", "http://docs.", "wiki", "tutorial", "guide")
        )

        if install and fenced and (howto or usage):
            qs = 0.94
        elif fenced and (install or howto or usage):
            qs = 0.79
        elif (install or howto or usage):
            qs = 0.62
        else:
            qs = 0.00

        if fenced and (notebook or demo):
            ex = 0.90
        elif fenced:
            ex = 0.79
        else:
            ex = 0.00

        if api and (cfg or trbl or docs_dir):
            dp = 0.87
        elif api or cfg or trbl or docs_dir:
            dp = 0.66
        else:
            dp = 0.00

        ed = 0.72 if ext else 0.00

        m = self._mh
        score = (
            m.quickstart * qs
            + m.examples * ex
            + m.depth * dp
            + m.ext * ed
        )
        return self._clip01(score)

    def _context_floor(self, ctx: Any) -> float:
        def has_src(objs: Any, src: str) -> bool:
            for o in (objs or []):
                origin = getattr(o, "__dict__", {}).get("_link_source")
                if origin == src:
                    return True
            return False

        prov = 0.0
        if ctx is not None:
            code = getattr(ctx, "linked_code", []) or []
            data = getattr(ctx, "linked_datasets", []) or []
            exp_code = has_src(code, "explicit")
            exp_data = has_src(data, "explicit")
            rd_code = has_src(code, "readme")
            rd_data = has_src(data, "readme")

            if exp_code and exp_data:
                prov = 0.90
            elif exp_code ^ exp_data:
                prov = 0.70
            elif rd_code or rd_data:
                prov = 0.50

        downloads = 0
        likes = 0
        if ctx is not None:
            downloads = int(getattr(ctx, "downloads_all_time", 0) or 0)
            likes = int(getattr(ctx, "likes", 0) or 0)

        if downloads > 1_000_000 or likes > 1_000:
            prov = max(prov, 0.90)
        elif downloads > 200_000 or likes > 200:
            prov = max(prov, 0.82)

        return self._clip01(prov)

    # ---- LLM wrapper ----
    def _llm_view(self, repo_context: dict, ctx: Any) -> Tuple[float, float, Dict[str, float]]:
        readme = self._compose_readme(repo_context, ctx)
        sys_msg = "Rate developer docs. Return ONLY JSON."
        prompt = (
            "Rate in [0,1] for: quickstart_clarity, examples_quality, "
            "docs_depth, external_docs_quality, setup_friction, confidence. "
            "Reward runnable, copy-paste steps and clear references.\n\n"
            "Return only JSON with those keys.\n\n"
            "README excerpt:\n---\n" + readme[:9000] + "\n---"
        )
        res = self._llm.ask_json(sys_msg, prompt, max_tokens=700)
        if not res.ok or not isinstance(res.data, dict):
            raise RuntimeError(res.error or "no JSON")

        def g(key: str) -> float:
            try:
                return self._clip01(float(res.data.get(key, 0.0)))
            except Exception:
                return 0.0

        parts = {
            "q": g("quickstart_clarity"),
            "e": g("examples_quality"),
            "d": g("docs_depth"),
            "x": g("external_docs_quality"),
            "fr": g("setup_friction"),
            "conf": g("confidence"),
        }
        parts["any_signal"] = any(parts[k] > 0.0 for k in ("q", "e", "d", "x"))

        m = self._mh
        doc = (
            m.quickstart * parts["q"]
            + m.examples * parts["e"]
            + m.depth * parts["d"]
            + m.ext * parts["x"]
        )
        doc = self._clip01(doc + 0.12 * parts["fr"])
        if doc > 0.0:
            doc = self._clip01(0.12 + 0.88 * doc)

        return doc, parts["conf"], parts

    # ---- utils ----
    @staticmethod
    def _bound(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    @staticmethod
    def _clip01(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v
