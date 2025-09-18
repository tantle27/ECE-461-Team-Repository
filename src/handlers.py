"""Thin URL handlers that build RepoContext objects using API clients."""

from pathlib import Path
import time
import requests
from api.hf_client import (
    HFClient,
    GatedRepoError,
    RepositoryNotFoundError,
    HfHubHTTPError,
)
from api.gh_client import GHClient
from repo_context import FileInfo, RepoContext
from url_router import UrlRouter, UrlType


class UrlHandler:
    """Base class for URL handlers."""

    def __init__(self, url: str = None):
        self.url = url

    def fetchMetaData(self) -> RepoContext:
        """Fetch metadata for the URL. Must be implemented by subclasses."""
        raise NotImplementedError(
            "fetchMetaData must be implemented by subclasses")


class ModelUrlHandler(UrlHandler):
    """Handler for Hugging Face model URLs."""

    def __init__(self, url: str = None):
        super().__init__(url)
        self.hf_client = HFClient()

    def fetchMetaData(self) -> RepoContext:
        """Fetch metadata for a Hugging Face model."""
        if not self.url:
            raise ValueError("URL is required")

        parsed = UrlRouter().parse(self.url)
        if parsed.type is not UrlType.MODEL or not parsed.hf_id:
            raise ValueError("URL is not a Hugging Face model URL")

        ctx = RepoContext(url=self.url, hf_id=parsed.hf_id, host="HF")

        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                info = self.hf_client.get_model_info(parsed.hf_id)
                ctx.card_data = info.card_data
                ctx.tags = info.tags
                ctx.downloads_30d = info.downloads_30d
                ctx.downloads_all_time = info.downloads_all_time
                ctx.likes = info.likes
                ctx.created_at = info.created_at
                ctx.last_modified = info.last_modified
                ctx.gated = info.gated
                ctx.private = info.private

                files = self.hf_client.list_files(
                    parsed.hf_id, repo_type="model")
                ctx.files = [
                    FileInfo(
                        path=Path(fi.path),
                        size_bytes=(fi.size or 0),
                        ext=Path(fi.path).suffix[1:].lower(),
                    )
                    for fi in files
                ]
                readme = self.hf_client.get_readme(parsed.hf_id)
                ctx.readme_text = readme
                model_idx = self.hf_client.get_model_index_json(parsed.hf_id)
                ctx.model_index = model_idx
                break

            except GatedRepoError as e:
                ctx.gated = True
                ctx.api_errors += 1
                ctx.fetch_logs.append(f"HF gated: {e}")
                break

            except RepositoryNotFoundError as e:
                ctx.api_errors += 1
                ctx.fetch_logs.append(f"HF not found: {e}")
                raise

            except HfHubHTTPError as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    # Rate limited - retry with backoff
                    msg = f"Rate limited, retrying in {retry_delay}s..."
                    ctx.fetch_logs.append(msg)
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    ctx.api_errors += 1
                    ctx.fetch_logs.append(f"HF HTTP error: {e}")
                    if "429" not in str(e):
                        raise
                    break

        return ctx


class DatasetUrlHandler(UrlHandler):
    """Handler for Hugging Face dataset URLs."""

    def __init__(self, url: str = None):
        super().__init__(url)
        self.hf_client = HFClient()

    def fetchMetaData(self) -> RepoContext:
        """Fetch metadata for a Hugging Face dataset."""
        if not self.url:
            raise ValueError("URL is required")

        parsed = UrlRouter().parse(self.url)
        if parsed.type is not UrlType.DATASET or not parsed.hf_id:
            raise ValueError("URL is not a Hugging Face dataset URL")

        ctx = RepoContext(url=self.url, hf_id=parsed.hf_id, host="HF")

        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                info = self.hf_client.get_dataset_info(parsed.hf_id)
                ctx.card_data = info.card_data
                ctx.tags = info.tags
                ctx.downloads_30d = info.downloads_30d
                ctx.downloads_all_time = info.downloads_all_time
                ctx.likes = info.likes
                ctx.created_at = info.created_at
                ctx.last_modified = info.last_modified
                ctx.gated = info.gated
                ctx.private = info.private

                ctx.readme_text = self.hf_client.get_readme(parsed.hf_id)
                break

            except GatedRepoError as e:
                ctx.gated = True
                ctx.api_errors += 1
                ctx.fetch_logs.append(f"HF gated: {e}")
                break

            except RepositoryNotFoundError as e:
                ctx.api_errors += 1
                ctx.fetch_logs.append(f"HF not found: {e}")
                raise

            except HfHubHTTPError as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    # Rate limited - retry with backoff
                    msg = f"Rate limited, retrying in {retry_delay}s..."
                    ctx.fetch_logs.append(msg)
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    ctx.api_errors += 1
                    ctx.fetch_logs.append(f"HF HTTP error: {e}")
                    if "429" not in str(e):
                        raise
                    break

        return ctx


class CodeUrlHandler(UrlHandler):
    """Handler for GitHub repository URLs."""

    def __init__(self, url: str = None):
        super().__init__(url)
        self.gh_client = GHClient()

    def fetchMetaData(self) -> RepoContext:
        """Fetch metadata for a GitHub repository."""
        if not self.url:
            raise ValueError("URL is required")

        parsed = UrlRouter().parse(self.url)
        if not parsed.gh_owner_repo:
            raise ValueError("URL is not a GitHub repository URL")

        owner, repo = parsed.gh_owner_repo

        # Use requests directly for GitHub API (simpler for testing)
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                url = f"https://api.github.com/repos/{owner}/{repo}"
                response = requests.get(url)

                if response.status_code == 404:
                    raise Exception(f"Repository not found: {owner}/{repo}")
                elif response.status_code == 403:
                    if attempt < max_retries - 1:
                        # Rate limited - retry with backoff
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        # Return empty context on rate limit
                        return RepoContext(
                            url=self.url,
                            gh_url=f"https://github.com/{owner}/{repo}",
                            host="GitHub",
                            api_errors=1,
                            fetch_logs=["GitHub API rate limited"],
                        )
                elif response.status_code == 200:
                    data = response.json()

                    return RepoContext(
                        url=self.url,
                        gh_url=f"https://github.com/{owner}/{repo}",
                        host="GitHub",
                        private=data.get("private", False),
                        readme_text=f"# {data.get('name', 'Repository')}",
                        contributors=[],  # Would need separate API call
                        created_at=data.get("created_at"),
                        last_modified=data.get("updated_at"),
                    )
                else:
                    error_msg = f"GitHub API error: {response.status_code}"
                    raise Exception(error_msg)

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise Exception(f"GitHub API request failed: {e}")


def build_model_context(url: str) -> RepoContext:
    handler = ModelUrlHandler(url)
    return handler.fetchMetaData()


def build_dataset_context(url: str) -> RepoContext:
    handler = DatasetUrlHandler(url)
    return handler.fetchMetaData()


def build_code_context(url: str) -> RepoContext:
    handler = CodeUrlHandler(url)
    return handler.fetchMetaData()
