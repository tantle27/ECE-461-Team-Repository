"""Thin URL handlers that build RepoContext objects using API clients."""

from pathlib import Path

from api.hf_client import (
    HFClient,
    GatedRepoError,
    RepositoryNotFoundError,
    HfHubHTTPError,
)
from api.gh_client import GHClient
from repo_context import FileInfo, RepoContext
from url_router import UrlRouter, UrlType


def build_model_context(url: str) -> RepoContext:
    parsed = UrlRouter().parse(url)
    if parsed.type is not UrlType.MODEL or not parsed.hf_id:
        raise ValueError("URL is not a Hugging Face model URL")

    hf = HFClient()
    ctx = RepoContext(url=url, hf_id=parsed.hf_id, host="HF")

    try:
        info = hf.get_model_info(parsed.hf_id)
        ctx.card_data = info.card_data
        ctx.tags = info.tags
        ctx.downloads_30d = info.downloads_30d
        ctx.downloads_all_time = info.downloads_all_time
        ctx.likes = info.likes
        ctx.created_at = info.created_at
        ctx.last_modified = info.last_modified
        ctx.gated = info.gated
        ctx.private = info.private

        files = hf.list_files(parsed.hf_id, repo_type="model")
        ctx.files = [
            FileInfo(
                path=Path(fi.path),
                size_bytes=(fi.size or 0),
                ext=Path(fi.path).suffix[1:].lower(),
            )
            for fi in files
        ]
        ctx.readme_text = hf.get_readme(parsed.hf_id)
        ctx.model_index = hf.get_model_index_json(parsed.hf_id)

    except GatedRepoError as e:
        ctx.gated = True
        ctx.api_errors += 1
        ctx.fetch_logs.append(f"HF gated: {e}")
        return ctx

    except RepositoryNotFoundError as e:
        ctx.api_errors += 1
        ctx.fetch_logs.append(f"HF not found: {e}")
        raise

    except HfHubHTTPError as e:
        ctx.api_errors += 1
        ctx.fetch_logs.append(f"HF HTTP error: {e}")
        return ctx


def build_dataset_context(url: str) -> RepoContext:
    parsed = UrlRouter().parse(url)
    if parsed.type is not UrlType.DATASET or not parsed.hf_id:
        raise ValueError("URL is not a Hugging Face dataset URL")

    hf = HFClient()
    ctx = RepoContext(url=url, hf_id=parsed.hf_id, host="HF")

    try:
        info = hf.get_dataset_info(parsed.hf_id)
        ctx.card_data = info.card_data
        ctx.tags = info.tags
        ctx.downloads_30d = info.downloads_30d
        ctx.downloads_all_time = info.downloads_all_time
        ctx.likes = info.likes
        ctx.created_at = info.created_at
        ctx.last_modified = info.last_modified
        ctx.gated = info.gated
        ctx.private = info.private

        ctx.readme_text = hf.get_readme(parsed.hf_id)

    except GatedRepoError as e:
        ctx.gated = True
        ctx.api_errors += 1
        ctx.fetch_logs.append(f"HF gated: {e}")
        return ctx
    except RepositoryNotFoundError as e:
        ctx.api_errors += 1
        ctx.fetch_logs.append(f"HF not found: {e}")
        raise
    except HfHubHTTPError as e:
        ctx.api_errors += 1
        ctx.fetch_logs.append(f"HF HTTP error: {e}")
        return ctx

    return ctx


def build_code_context(url: str) -> RepoContext:
    """Build a RepoContext for a GitHub repository URL."""
    parsed = UrlRouter().parse(url)
    if not parsed.gh_owner_repo:
        raise ValueError("URL is not a GitHub repository URL")

    owner, repo = parsed.gh_owner_repo
    gh = GHClient()
    info = gh.get_repo(owner, repo)
    readme = gh.get_readme_markdown(owner, repo)
    contributors = gh.list_contributors(owner, repo)

    return RepoContext(
        url=url,
        gh_url=f"https://github.com/{owner}/{repo}",
        host="GitHub",
        readme_text=readme,
        contributors=contributors,
        private=(info.private if info else None),
    )
