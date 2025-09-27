# metrics/ramp_up_time_metric.py

@dataclass(frozen=True)
class HeuristicWeights:
    quickstart: float = 0.30
    examples: float = 0.30
    docs_depth: float = 0.25
    ext_docs: float = 0.15

@dataclass(frozen=True)
class BlendCfg:
    llm_weight: float = 0.70
    heu_weight: float = 0.30
    fairness_cap: float = 0.45  # allow a bit more swing

class RampUpTimeMetric(BaseMetric):
    def __init__(self, weight: float = 0.15, use_llm: bool = True):
        super().__init__(name="RampUpTime", weight=weight)
        self._use_llm, self._llm = use_llm, LLMClient()
        self._hw, self._blend = HeuristicWeights(), BlendCfg()

    # ---------- public API ----------

    def evaluate(self, repo_context: dict) -> float:
        ctx = repo_context.get("_ctx_obj")  # no isinstance check
        readme = self._gather_readme(repo_context, ctx)
        heur = self._combine_heuristics(**self._heuristic_parts(readme))

        prov_floor = self._provenance_floor(ctx)
        heur = max(heur, prov_floor)

        if not (self._use_llm and getattr(self._llm, "provider", None)):
            return heur

        try:
            llm, conf, parts = self._score_with_llm(readme)
            repo_context["_rampup_llm_parts"] = parts

            base_w = self._blend.llm_weight
            conf_w = min(0.25, 0.5 * max(0.0, conf - 0.5))
            w_llm = min(0.9, base_w + conf_w)
            print("RampUpTime LLM weight: base %.2f + conf %.2f = %.2f" % (base_w, conf_w, w_llm))
            w_heu = 1.0 - w_llm
            raw = w_llm * llm + w_heu * heur

            cap = self._blend.fairness_cap
            final = heur + max(-cap, min(cap, raw - heur))

            if parts.get("any_signal"):
                final = max(final, max(0.25, prov_floor))

            return self._clamp01(final)
        except Exception:
            return heur

    # ---------- provenance floor ----------

    def _provenance_floor(self, ctx: Optional[RepoContext]) -> float:
        if not isinstance(ctx, RepoContext):
            print("No RepoContext for provenance floor")
            return 0.0

        def has(objs, src):  # check _link_source provenance tag
            for o in objs or []:
                if getattr(o, "__dict__", {}).get("_link_source") == src:
                    return True
            return False

        explicit_code = has(getattr(ctx, "linked_code", []), "explicit")
        explicit_ds   = has(getattr(ctx, "linked_datasets", []), "explicit")
        readme_code   = has(getattr(ctx, "linked_code", []), "readme")
        readme_ds     = has(getattr(ctx, "linked_datasets", []), "readme")

        if explicit_code and explicit_ds:
            return 0.85
        if explicit_code ^ explicit_ds:
            return 0.60
        if readme_code or readme_ds:
            return 0.40
        return 0.0

    # ---------- Heuristics (more generous & broader coverage) ----------

    def _gather_readme(self, repo_context: dict, ctx: Optional[RepoContext]) -> str:
        base = (repo_context.get("readme_text") or "").strip()
        if isinstance(ctx, RepoContext) and ctx.linked_code:
            best = max(
                (c for c in ctx.linked_code if isinstance(c, RepoContext)),
                key=lambda c: len(c.readme_text or ""),
                default=None,
            )
            if best and best.readme_text:
                extra = best.readme_text.strip()[:6000]
                if extra and extra not in base:
                    base = f"{base}\n\n---\n# Linked code README\n{extra}"
        return base

    def _heuristic_parts(self, readme: str) -> Dict[str, float]:
        r = (readme or "").lower()

        has_install = any(k in r for k in (
            "pip install", "conda install", "poetry add",
            "pip3 install", "requirements.txt"
        ))
        has_quick = any(k in r for k in (
            "getting started", "quickstart", "quick start", "setup", "how to"
        ))
        has_usage = any(k in r for k in (
            "usage", "example", "inference", "train", "run", "predict", "fine-tune", "finetune"
        ))
        has_code_fence = "```" in r or "$ " in r
        # quickstart: reward install or copy-pastable code + usage
        quickstart = (
            0.90 if (has_install and has_code_fence and (has_quick or has_usage)) else
            0.70 if ((has_install or has_quick) and has_code_fence) else
            0.55 if (has_install or has_quick or has_usage) else
            0.0
        )

        # examples: any runnable snippet is good; notebook/colab/demo makes it excellent
        has_notebook = any(k in r for k in ("colab", ".ipynb", "notebook"))
        has_demo = any(k in r for k in ("demo", "sample", "samples", "examples"))
        examples = (
            0.90 if (has_code_fence and (has_notebook or has_demo)) else
            0.75 if has_code_fence else
            0.0
        )

        # docs depth: look for reference, configuration, FAQ, parameters, options, docs/ folder, etc.
        has_api = any(k in r for k in ("api reference", "api docs", "reference", "documentation"))
        has_cfg = any(k in r for k in ("configuration", "config", "settings", "parameters", "options"))
        has_trb = any(k in r for k in ("troubleshooting", "faq", "known issues"))
        has_docs_dir = "docs/" in r or "mkdocs" in r or "sphinx" in r
        docs_depth = (
            0.88 if (has_api and (has_cfg or has_trb or has_docs_dir)) else
            0.65 if (has_api or has_cfg or has_trb or has_docs_dir) else
            0.0
        )

        # external docs: readthedocs, hosted docs, wiki, tutorial site, blog guide
        has_ext = any(k in r for k in (
            "readthedocs", "https://docs.", "http://docs.", "wiki", "mkdocs",
            "tutorial", "guide", "blog", "arxiv.org"
        ))
        ext_docs = 0.65 if has_ext else 0.0

        return {
            "quickstart": quickstart,
            "examples": examples,
            "docs_depth": docs_depth,
            "ext_docs": ext_docs,
        }

    def _combine_heuristics(self, *, quickstart: float, examples: float, docs_depth: float, ext_docs: float) -> float:
        hw = self._hw
        score = (
            hw.quickstart * self._clamp01(quickstart) +
            hw.examples   * self._clamp01(examples)   +
            hw.docs_depth * self._clamp01(docs_depth) +
            hw.ext_docs   * self._clamp01(ext_docs)
        )
        return self._clamp01(score)

    @staticmethod
    def _clamp01(x: Any) -> float:
        try:
            xf = float(x)
        except Exception:
            return 0.0
        return 0.0 if xf < 0.0 else 1.0 if xf > 1.0 else xf