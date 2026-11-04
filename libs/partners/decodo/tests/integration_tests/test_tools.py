"""Integration tests for langchain_decodo tools.

Run with a real Decodo API key in DECODO_API_TOKEN:

    DECODO_API_TOKEN=your_api_key pytest tests/integration_tests/

To run with a basic auth token from an older plan instead, also set
DECODO_AUTH_MODE=basic. The search, map and crawl tests are skipped then, since
those endpoints only accept an API key.

All tests are skipped automatically when ``DECODO_API_TOKEN`` is not set in
the environment, so this file is safe to include in CI without secrets.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import pytest

# Skip the entire module if no credential is configured.
pytestmark = pytest.mark.skipif(
    not os.environ.get("DECODO_API_TOKEN"),
    reason="DECODO_API_TOKEN environment variable (API key) is not set",
)

AUTH_MODE: Literal["basic", "token"] = (
    "basic" if os.environ.get("DECODO_AUTH_MODE") == "basic" else "token"
)


@pytest.fixture()
def scrape_tool() -> Any:
    from langchain_decodo.tools import DecodoWebScrapeTool

    return DecodoWebScrapeTool(auth_mode=AUTH_MODE)


@pytest.fixture()
def search_tool() -> Any:
    from langchain_decodo.tools import DecodoSearchTool

    return DecodoSearchTool(auth_mode=AUTH_MODE)


@pytest.fixture()
def loader() -> Any:
    from langchain_decodo import DecodoLoader

    return DecodoLoader(urls=["https://example.com"], auth_mode=AUTH_MODE)


# ---------------------------------------------------------------------------
# DecodoWebScrapeTool integration tests
# ---------------------------------------------------------------------------


class TestDecodoWebScrapeToolIntegration:
    def test_scrape_example_dot_com(self, scrape_tool: object) -> None:
        from langchain_decodo.tools import DecodoWebScrapeTool

        assert isinstance(scrape_tool, DecodoWebScrapeTool)
        result = scrape_tool._run("https://example.com")
        assert isinstance(result, str)
        assert len(result) > 0
        # example.com always contains "Example Domain"
        assert "Example Domain" in result or "example" in result.lower()

    def test_scrape_returns_string(self, scrape_tool: object) -> None:
        from langchain_decodo.tools import DecodoWebScrapeTool

        assert isinstance(scrape_tool, DecodoWebScrapeTool)
        result = scrape_tool._run("https://httpbin.org/html")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# DecodoSearchTool integration tests
# ---------------------------------------------------------------------------


class TestDecodoSearchToolIntegration:
    def test_google_search_returns_results(self, search_tool: object) -> None:
        from langchain_decodo.tools import DecodoSearchTool

        assert isinstance(search_tool, DecodoSearchTool)
        raw = search_tool._run("Python programming language", engine="google", num_results=3)
        results = json.loads(raw)
        assert isinstance(results, list)
        assert len(results) > 0
        # Each result should have a content field
        for item in results:
            assert "content" in item

    def test_amazon_search_returns_results(self, search_tool: object) -> None:
        from langchain_decodo.tools import DecodoSearchTool

        assert isinstance(search_tool, DecodoSearchTool)
        raw = search_tool._run("Python book", engine="amazon", num_results=3)
        results = json.loads(raw)
        assert isinstance(results, list)

    def test_reddit_search_prepends_filter(self, search_tool: object) -> None:
        from langchain_decodo.tools import DecodoSearchTool

        assert isinstance(search_tool, DecodoSearchTool)
        # Should not raise; results may be empty depending on query
        raw = search_tool._run("best Python libraries", engine="reddit", num_results=3)
        results = json.loads(raw)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# DecodoLoader integration tests
# ---------------------------------------------------------------------------


class TestDecodoLoaderIntegration:
    def test_load_single_url(self) -> None:
        from langchain_core.documents import Document

        from langchain_decodo import DecodoLoader

        loader = DecodoLoader(urls="https://example.com", auth_mode=AUTH_MODE)
        docs = loader.load()
        assert isinstance(docs, list)
        assert len(docs) > 0
        doc = docs[0]
        assert isinstance(doc, Document)
        assert doc.metadata.get("url") == "https://example.com"
        assert doc.metadata.get("source") == "https://example.com"

    def test_lazy_load_yields_documents(self) -> None:
        from langchain_core.documents import Document

        from langchain_decodo import DecodoLoader

        loader = DecodoLoader(urls=["https://example.com"], auth_mode=AUTH_MODE)
        docs = list(loader.lazy_load())
        assert all(isinstance(d, Document) for d in docs)

    def test_continue_on_error_skips_bad_url(self) -> None:
        from langchain_decodo import DecodoLoader

        loader = DecodoLoader(
            urls=["https://this-domain-definitely-does-not-exist-xyz123.com"],
            auth_mode=AUTH_MODE,
            continue_on_error=True,
        )
        docs = loader.load()
        # Should not raise; yields empty doc with error metadata
        assert isinstance(docs, list)
        if docs:
            assert "error" in docs[0].metadata


# ---------------------------------------------------------------------------
# Search, map and crawl tools (API key only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(AUTH_MODE == "basic", reason="search, map and crawl need an API key")
class TestDataApiToolsIntegration:
    def test_web_search_returns_organic_results(self) -> None:
        from langchain_decodo import DecodoWebSearchTool

        result = json.loads(DecodoWebSearchTool().invoke({"query": "Python programming language"}))
        assert len(result["organic"]) > 0
        assert "link" in result["organic"][0]

    def test_map_lists_links(self) -> None:
        from langchain_decodo import DecodoMapTool

        result = json.loads(DecodoMapTool().invoke({"url": "https://www.python.org", "limit": 5}))
        links = result["results"][0]["content"]["links"]
        assert 0 < len(links) <= 5

    def test_crawl_returns_page_content(self) -> None:
        from langchain_decodo import DecodoCrawlTool

        tool = DecodoCrawlTool(timeout=300)
        result = json.loads(tool.invoke({"url": "https://example.com", "limit": 1}))
        assert result["status"]["status"] in {"done", "truncated"}
        assert len(result["pages"]) > 0
        assert "documentation examples" in result["pages"][0].get("content", "")
