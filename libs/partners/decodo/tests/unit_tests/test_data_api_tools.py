"""Unit tests for the search, map and crawl tools.

The tools go through the real `decodo` SDK; only `httpx.request` inside the SDK
is replaced, so these tests also check the URLs, headers and bodies it sends.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from langchain_decodo import (
    DecodoCrawlCancelTool,
    DecodoCrawlTool,
    DecodoMapTool,
    DecodoWebSearchTool,
)
from langchain_decodo.data_api_tools import _DecodoDataApiTool

API_KEY = SecretStr("api-key")
BASE_URL = "https://data.decodo.com"


class FakeApi:
    """Answers SDK requests from a queue of responses per `METHOD path`."""

    def __init__(self) -> None:
        self.routes: dict[str, list[tuple[int, Any]]] = {}
        self.calls: list[dict[str, Any]] = []

    def add(self, route: str, body: Any, status_code: int = 200) -> None:
        self.routes.setdefault(route, []).append((status_code, body))

    def request(self, method: str, url: str, **kwargs: Any) -> MagicMock:
        path = url.removeprefix(BASE_URL)
        self.calls.append({"method": method, "url": url, "path": path, **kwargs})
        queue = self.routes[f"{method} {path.split('?')[0]}"]
        status_code, body = queue.pop(0) if len(queue) > 1 else queue[0]
        response = MagicMock()
        response.status_code = status_code
        response.is_success = 200 <= status_code < 300
        response.json.return_value = body
        return response

    def bodies(self, route: str) -> list[Any]:
        method, path = route.split(" ", 1)
        return [c["json"] for c in self.calls if c["method"] == method and c["path"] == path]


@pytest.fixture()
def api() -> Iterator[FakeApi]:
    fake = FakeApi()
    with patch("decodo.http.httpx.request", side_effect=fake.request):
        yield fake


@pytest.fixture()
def no_sleep() -> Iterator[MagicMock]:
    with patch("decodo.api.web_scraping_api.time.sleep") as sleep:
        yield sleep


ALL_TOOLS: list[Any] = [
    DecodoWebSearchTool,
    DecodoMapTool,
    DecodoCrawlTool,
    DecodoCrawlCancelTool,
]


# ---------------------------------------------------------------------------
# Shared behavior
# ---------------------------------------------------------------------------


class TestAuthentication:
    @pytest.mark.parametrize("tool_cls", ALL_TOOLS)
    def test_reads_env_api_key(self, tool_cls: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DECODO_API_TOKEN", "env-key")
        tool: _DecodoDataApiTool = tool_cls()
        assert tool.decodo_api_token.get_secret_value() == "env-key"

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DECODO_API_TOKEN", raising=False)
        with pytest.raises(ValueError, match="Web Data API key is required"):
            DecodoWebSearchTool()._run("coffee")

    def test_whitespace_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Web Data API key is required"):
            DecodoWebSearchTool(decodo_api_token=SecretStr("  "))._run("coffee")

    def test_sends_bearer_key_and_integration_header_to_data_api(self, api: FakeApi) -> None:
        api.add("POST /v1/search", {"organic": []})
        DecodoWebSearchTool(decodo_api_token=API_KEY)._run("coffee")

        call = api.calls[0]
        assert call["url"] == f"{BASE_URL}/v1/search"
        assert call["headers"]["Authorization"] == "Bearer api-key"
        assert call["headers"]["x-integration"] == "langchain-python"

    def test_rejected_key_explains_basic_tokens_do_not_work(self, api: FakeApi) -> None:
        api.add("POST /v1/search", {"message": "Unauthorized"}, status_code=401)
        tool = DecodoWebSearchTool(decodo_api_token=SecretStr("base64-user-pass"))
        with pytest.raises(RuntimeError, match="Basic authentication tokens from older plans"):
            tool._run("coffee")


class TestErrors:
    def test_api_error_becomes_runtime_error(self, api: FakeApi) -> None:
        api.add("POST /v1/map", {"message": "Internal error"}, status_code=500)
        with pytest.raises(RuntimeError, match="Decodo API error: Internal error"):
            DecodoMapTool(decodo_api_token=API_KEY)._run("https://example.com")

    def test_network_error_becomes_runtime_error(self) -> None:
        with patch("decodo.http.httpx.request", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(RuntimeError, match="network error: refused"):
                DecodoMapTool(decodo_api_token=API_KEY)._run("https://example.com")

    def test_non_json_success_becomes_runtime_error(self) -> None:
        response = MagicMock(status_code=200, is_success=True)
        response.json.side_effect = json.JSONDecodeError("Expecting value", "<html>", 0)
        with patch("decodo.http.httpx.request", return_value=response):
            with pytest.raises(RuntimeError, match="not JSON"):
                DecodoMapTool(decodo_api_token=API_KEY)._run("https://example.com")

    def test_rejected_key_does_not_blame_basic_auth_alone(self, api: FakeApi) -> None:
        api.add("POST /v1/map", {"message": "Forbidden"}, status_code=403)
        with pytest.raises(RuntimeError, match="valid and that its plan includes this endpoint"):
            DecodoMapTool(decodo_api_token=API_KEY)._run("https://example.com")

    def test_timeout_becomes_runtime_error(self) -> None:
        with patch("decodo.http.httpx.request", side_effect=httpx.ReadTimeout("slow")):
            with pytest.raises(RuntimeError, match="timed out"):
                DecodoMapTool(decodo_api_token=API_KEY)._run("https://example.com")


# ---------------------------------------------------------------------------
# DecodoWebSearchTool
# ---------------------------------------------------------------------------


class TestDecodoWebSearchTool:
    def test_returns_search_document(self, api: FakeApi) -> None:
        document = {"organic": [{"rank": 1, "title": "Cafe", "link": "https://cafe.example"}]}
        api.add("POST /v1/search", document)
        result = DecodoWebSearchTool(decodo_api_token=API_KEY)._run("coffee")
        assert json.loads(result) == document

    def test_sends_only_given_fields(self, api: FakeApi) -> None:
        api.add("POST /v1/search", {"organic": []})
        tool = DecodoWebSearchTool(decodo_api_token=API_KEY)
        tool._run("coffee")
        tool._run("coffee", geo="United States", locale="en-US", page=2)
        assert api.bodies("POST /v1/search") == [
            {"query": "coffee"},
            {"query": "coffee", "geo": "United States", "locale": "en-US", "page": 2},
        ]

    def test_failed_search_raises(self, api: FakeApi) -> None:
        api.add("POST /v1/search", {"status": "failed", "message": "upstream blocked"})
        with pytest.raises(RuntimeError, match="Decodo search failed: upstream blocked"):
            DecodoWebSearchTool(decodo_api_token=API_KEY)._run("coffee")

    def test_name_differs_from_scrape_based_search_tool(self) -> None:
        assert DecodoWebSearchTool(decodo_api_token=API_KEY).name == "decodo_web_search"

    def test_rejects_page_below_one(self) -> None:
        tool = DecodoWebSearchTool(decodo_api_token=API_KEY)
        with pytest.raises(ValidationError):
            tool.invoke({"query": "coffee", "page": 0})


# ---------------------------------------------------------------------------
# DecodoMapTool
# ---------------------------------------------------------------------------


MAP_RESPONSE = {
    "results": [
        {
            "content": {"links": [{"url": "https://example.com/a"}]},
            "status_code": 200,
            "url": "https://example.com",
            "sitemap_status": "include",
        }
    ]
}


class TestDecodoMapTool:
    def test_returns_map_response(self, api: FakeApi) -> None:
        api.add("POST /v1/map", MAP_RESPONSE)
        result = DecodoMapTool(decodo_api_token=API_KEY)._run("https://example.com")
        assert json.loads(result) == MAP_RESPONSE

    def test_default_payload(self, api: FakeApi) -> None:
        api.add("POST /v1/map", MAP_RESPONSE)
        DecodoMapTool(decodo_api_token=API_KEY).invoke({"url": "https://example.com"})
        assert api.bodies("POST /v1/map") == [{"url": "https://example.com", "limit": 100}]

    def test_maps_arguments_to_api_fields(self, api: FakeApi) -> None:
        api.add("POST /v1/map", MAP_RESPONSE)
        DecodoMapTool(decodo_api_token=API_KEY).invoke(
            {
                "url": "https://example.com",
                "sitemap": "skip",
                "domain_filter": "subdomain",
                "select_paths": ["^/blog/"],
                "exclude_paths": ["^/tag/"],
                "geo": "Germany",
                "locale": "de-DE",
                "limit": 50,
                "js_render": True,
            }
        )
        assert api.bodies("POST /v1/map") == [
            {
                "url": "https://example.com",
                "sitemap": "skip",
                "domain_filter": "subdomain",
                "select_paths": ["^/blog/"],
                "exclude_paths": ["^/tag/"],
                "geo": "Germany",
                "locale": "de-DE",
                "limit": 50,
                "JS_rendering": True,
            }
        ]

    @pytest.mark.parametrize(
        "response",
        [
            {"results": [{**MAP_RESPONSE["results"][0], "status_code": 403}]},
            {"results": [], "message": "blocked"},
        ],
    )
    def test_failed_map_raises(self, api: FakeApi, response: dict[str, Any]) -> None:
        api.add("POST /v1/map", response)
        with pytest.raises(RuntimeError, match="Decodo map failed"):
            DecodoMapTool(decodo_api_token=API_KEY)._run("https://example.com")

    def test_rejects_limit_above_max(self) -> None:
        with pytest.raises(ValidationError):
            DecodoMapTool(decodo_api_token=API_KEY).invoke(
                {"url": "https://example.com", "limit": 501}
            )


# ---------------------------------------------------------------------------
# DecodoCrawlTool
# ---------------------------------------------------------------------------

CRAWL_ID = "crawl-123"
STARTED = {"crawl_id": CRAWL_ID, "status": "pending"}
CRAWL_STATUS_LINK = f"{BASE_URL}/v1/crawl/{CRAWL_ID}/status"
EXPECTED_PAGES = [
    {
        "url": "https://example.com",
        "content": "# Home",
        "status": "done",
        "depth": 0,
        "crawl_status": CRAWL_STATUS_LINK,
    },
    {
        "url": "https://example.com/x",
        "status": "faulted",
        "depth": 1,
        "crawl_status": CRAWL_STATUS_LINK,
    },
]
PAGES = {"crawl_id": CRAWL_ID, "url": EXPECTED_PAGES}


def _status(state: str, **extra: Any) -> dict[str, Any]:
    return {
        "crawl_id": CRAWL_ID,
        "status": state,
        "discovered": 2,
        "completed": 1,
        "failed": 1,
        "in_flight": 0,
        **extra,
    }


class TestDecodoCrawlTool:
    def test_polls_until_done_and_returns_pages(self, api: FakeApi, no_sleep: MagicMock) -> None:
        api.add("POST /v1/crawl", STARTED)
        api.add(f"GET /v1/crawl/{CRAWL_ID}/status", _status("pending"))
        api.add(f"GET /v1/crawl/{CRAWL_ID}/status", _status("done"))
        api.add(f"GET /v1/crawl/{CRAWL_ID}", PAGES)

        result = json.loads(
            DecodoCrawlTool(decodo_api_token=API_KEY, poll_interval=2)._run("https://example.com")
        )

        assert result == {"status": _status("done"), "pages": EXPECTED_PAGES}
        no_sleep.assert_called_once_with(2)
        results_call = next(c for c in api.calls if c["path"].startswith(f"/v1/crawl/{CRAWL_ID}?"))
        assert results_call["path"] == f"/v1/crawl/{CRAWL_ID}?type=content"

    def test_follows_results_pagination(self, api: FakeApi, no_sleep: MagicMock) -> None:
        api.add("POST /v1/crawl", STARTED)
        api.add(f"GET /v1/crawl/{CRAWL_ID}/status", _status("done"))
        api.add(
            f"GET /v1/crawl/{CRAWL_ID}",
            {"url": [PAGES["url"][0]], "next": f"{BASE_URL}/v1/crawl/{CRAWL_ID}?cursor=2"},
        )
        api.add(f"GET /v1/crawl/{CRAWL_ID}", {"url": [PAGES["url"][1]]})

        result = json.loads(DecodoCrawlTool(decodo_api_token=API_KEY)._run("https://example.com"))
        assert result["pages"] == EXPECTED_PAGES

    @pytest.mark.parametrize("state", ["truncated", "faulted", "cancelled"])
    def test_early_stop_returns_status_and_collected_pages(
        self, api: FakeApi, no_sleep: MagicMock, state: str
    ) -> None:
        api.add("POST /v1/crawl", STARTED)
        api.add(f"GET /v1/crawl/{CRAWL_ID}/status", _status(state, reason="vendor_limit"))
        api.add(f"GET /v1/crawl/{CRAWL_ID}", PAGES)

        result = json.loads(DecodoCrawlTool(decodo_api_token=API_KEY)._run("https://example.com"))
        assert result == {
            "status": _status(state, reason="vendor_limit"),
            "pages": EXPECTED_PAGES,
        }

    def test_timeout_returns_pending_status_and_pages_so_far(
        self, api: FakeApi, no_sleep: MagicMock
    ) -> None:
        api.add("POST /v1/crawl", STARTED)
        api.add(f"GET /v1/crawl/{CRAWL_ID}/status", _status("pending"))
        api.add(f"GET /v1/crawl/{CRAWL_ID}", PAGES)
        tool = DecodoCrawlTool(decodo_api_token=API_KEY, timeout=10)

        with patch("decodo.api.web_scraping_api.time.monotonic", side_effect=[0, 4, 11]):
            result = json.loads(tool._run("https://example.com"))

        assert result == {"status": _status("pending"), "pages": EXPECTED_PAGES}
        no_sleep.assert_called_once_with(5)

    def test_no_pages_returns_null_status(self, api: FakeApi, no_sleep: MagicMock) -> None:
        api.add("POST /v1/crawl", STARTED)
        api.add(f"GET /v1/crawl/{CRAWL_ID}/status", _status("truncated", reason="no_pages_found"))
        api.add(f"GET /v1/crawl/{CRAWL_ID}", {"crawl_id": CRAWL_ID, "url": []})

        result = json.loads(DecodoCrawlTool(decodo_api_token=API_KEY)._run("https://example.com"))
        assert result == {"status": None, "pages": []}

    def test_unrecognized_status_link_returns_null_status(
        self, api: FakeApi, no_sleep: MagicMock
    ) -> None:
        pages = [
            {**page, "crawl_status": f"{BASE_URL}/v1/crawl/{CRAWL_ID}"} for page in EXPECTED_PAGES
        ]
        api.add("POST /v1/crawl", STARTED)
        api.add(f"GET /v1/crawl/{CRAWL_ID}/status", _status("done"))
        api.add(f"GET /v1/crawl/{CRAWL_ID}", {"url": pages})

        result = json.loads(DecodoCrawlTool(decodo_api_token=API_KEY)._run("https://example.com"))
        assert result == {"status": None, "pages": pages}

    def test_failed_final_status_keeps_pages(self, api: FakeApi, no_sleep: MagicMock) -> None:
        status_route = f"GET /v1/crawl/{CRAWL_ID}/status"
        api.add("POST /v1/crawl", STARTED)
        api.add(status_route, _status("done"))
        api.add(status_route, {"message": "Internal error"}, status_code=500)
        api.add(f"GET /v1/crawl/{CRAWL_ID}", PAGES)

        result = json.loads(DecodoCrawlTool(decodo_api_token=API_KEY)._run("https://example.com"))
        assert result == {"status": None, "pages": EXPECTED_PAGES}

    def test_results_failure_after_early_stop_names_crawl(
        self, api: FakeApi, no_sleep: MagicMock
    ) -> None:
        api.add("POST /v1/crawl", STARTED)
        api.add(f"GET /v1/crawl/{CRAWL_ID}/status", _status("faulted"))
        api.add(f"GET /v1/crawl/{CRAWL_ID}", {"message": "Internal error"}, status_code=500)

        with pytest.raises(RuntimeError, match=f"^Crawl {CRAWL_ID}: Decodo API error"):
            DecodoCrawlTool(decodo_api_token=API_KEY)._run("https://example.com")

    @pytest.mark.parametrize("failing_route", ["status", "results"])
    def test_failure_after_start_names_crawl(
        self, api: FakeApi, no_sleep: MagicMock, failing_route: str
    ) -> None:
        api.add("POST /v1/crawl", STARTED)
        status_route = f"GET /v1/crawl/{CRAWL_ID}/status"
        results_route = f"GET /v1/crawl/{CRAWL_ID}"
        if failing_route == "status":
            api.add(status_route, {"message": "Too many requests"}, status_code=429)
        else:
            api.add(status_route, _status("done"))
            api.add(results_route, {"message": "Internal error"}, status_code=500)

        with pytest.raises(RuntimeError, match=f"Crawl {CRAWL_ID}"):
            DecodoCrawlTool(decodo_api_token=API_KEY)._run("https://example.com")

    def test_default_payload_sends_timeout_as_max_duration(
        self, api: FakeApi, no_sleep: MagicMock
    ) -> None:
        api.add("POST /v1/crawl", STARTED)
        api.add(f"GET /v1/crawl/{CRAWL_ID}/status", _status("done"))
        api.add(f"GET /v1/crawl/{CRAWL_ID}", PAGES)

        DecodoCrawlTool(decodo_api_token=API_KEY).invoke({"url": "https://example.com"})
        assert api.bodies("POST /v1/crawl") == [
            {"url": "https://example.com", "limit": 10, "output": "markdown", "max_duration": 600}
        ]

    def test_maps_arguments_to_api_fields(self, api: FakeApi, no_sleep: MagicMock) -> None:
        api.add("POST /v1/crawl", STARTED)
        api.add(f"GET /v1/crawl/{CRAWL_ID}/status", _status("done"))
        api.add(f"GET /v1/crawl/{CRAWL_ID}", PAGES)

        DecodoCrawlTool(decodo_api_token=API_KEY, timeout=90.5).invoke(
            {
                "url": "https://example.com",
                "domain_filter": "domain",
                "select_paths": ["^/docs/"],
                "limit": 25,
                "max_depth": 3,
                "output": "html",
            }
        )
        assert api.bodies("POST /v1/crawl") == [
            {
                "url": "https://example.com",
                "domain_filter": "domain",
                "select_paths": ["^/docs/"],
                "limit": 25,
                "output": "html",
                "max_depth": 3,
                "max_duration": 91,
            }
        ]

    def test_caps_max_duration_at_api_ceiling(self, api: FakeApi, no_sleep: MagicMock) -> None:
        api.add("POST /v1/crawl", STARTED)
        api.add(f"GET /v1/crawl/{CRAWL_ID}/status", _status("done"))
        api.add(f"GET /v1/crawl/{CRAWL_ID}", PAGES)

        DecodoCrawlTool(decodo_api_token=API_KEY, timeout=5 * 60 * 60)._run("https://example.com")
        assert api.bodies("POST /v1/crawl")[0]["max_duration"] == 4 * 60 * 60

    @pytest.mark.parametrize("field", ["poll_interval", "timeout"])
    def test_rejects_non_positive_durations(self, field: str) -> None:
        with pytest.raises(ValidationError):
            DecodoCrawlTool.model_validate({"decodo_api_token": API_KEY, field: 0})

    def test_rejects_limit_above_max(self) -> None:
        with pytest.raises(ValidationError):
            DecodoCrawlTool(decodo_api_token=API_KEY).invoke(
                {"url": "https://example.com", "limit": 101}
            )


# ---------------------------------------------------------------------------
# DecodoCrawlCancelTool
# ---------------------------------------------------------------------------


class TestDecodoCrawlCancelTool:
    def test_cancels_crawl(self, api: FakeApi) -> None:
        cancelled = {"crawl_id": CRAWL_ID, "status": "cancelled"}
        api.add(f"POST /v1/crawl/{CRAWL_ID}/cancel", cancelled)

        result = DecodoCrawlCancelTool(decodo_api_token=API_KEY).invoke({"crawl_id": CRAWL_ID})

        assert json.loads(result) == cancelled
        assert api.calls[0]["url"] == f"{BASE_URL}/v1/crawl/{CRAWL_ID}/cancel"
