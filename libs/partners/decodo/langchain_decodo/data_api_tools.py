"""LangChain tools for the Decodo Web Data API search, map and crawl endpoints.

- `DecodoWebSearchTool`: Runs a web search and returns ranked results.
- `DecodoMapTool`: Lists the URLs on a website without fetching their content.
- `DecodoCrawlTool`: Crawls a website and returns the content of every page found.
- `DecodoCrawlCancelTool`: Stops a crawl that `DecodoCrawlTool` stopped waiting for.

These endpoints exist only on `https://data.decodo.com`, which accepts API keys
and not the basic authentication tokens of older plans, so every tool here needs
an API key.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from typing import Any, Literal, TypeVar

import httpx
from decodo import (
    AuthenticationError,
    CrawlError,
    CrawlPage,
    CrawlStatusResponse,
    CrawlTimeoutError,
    DecodoClient,
    DecodoConfig,
    DecodoError,
    WebScrapingApi,
    WebScrapingApiConfig,
)
from decodo import TimeoutError as DecodoTimeoutError
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr, model_validator

from langchain_decodo._common import INTEGRATION_HEADER, is_failed, token_from_env

_MAP_LIMIT_MAX = 500
_MAP_DEFAULT_LIMIT = 100
_CRAWL_LIMIT_MAX = 100
_CRAWL_DEFAULT_LIMIT = 10
_CRAWL_MAX_DEPTH_MAX = 10_000

_DEFAULT_POLL_INTERVAL = 5.0  # seconds
_DEFAULT_CRAWL_TIMEOUT = 600.0  # seconds
# The API rejects a `max_duration` above 4 hours.
_MAX_DURATION_CEILING = 4 * 60 * 60  # seconds
_CRAWL_STATUS_LINK = re.compile(r"/v1/crawl/([^/?#]+)/status$")

_T = TypeVar("_T")


def _call_api(call: Callable[[], _T], *, context: str = "") -> _T:
    """Run an SDK call and re-raise its errors as `RuntimeError`, like the scrape tools.

    Args:
        call: Zero-argument callable that makes the SDK request.
        context: Prefix for the error message, such as the crawl the call was for.

    Returns:
        Whatever `call` returns.

    Raises:
        RuntimeError: On an API error, a timeout, a network failure or a
            response that is not JSON.
    """
    try:
        return call()
    except AuthenticationError as exc:
        msg = (
            f"Decodo API error: {exc}. Check that the API key is valid and that its plan "
            "includes this endpoint. Basic authentication tokens from older plans do not "
            "work here."
        )
        reason: Exception = exc
    except DecodoError as exc:
        msg, reason = f"Decodo API error: {exc}", exc
    except DecodoTimeoutError as exc:
        msg, reason = f"Decodo API request timed out: {exc}", exc
    # The SDK lets connection errors and unparsable bodies through as they are.
    except httpx.RequestError as exc:
        msg, reason = f"Decodo API network error: {exc}", exc
    except json.JSONDecodeError as exc:
        msg, reason = f"Decodo API returned a response that is not JSON: {exc}", exc
    raise RuntimeError(f"{context}{msg}") from reason


def _site_discovery_params(
    url: str,
    sitemap: str | None,
    domain_filter: str | None,
    select_paths: list[str] | None,
    exclude_paths: list[str] | None,
    geo: str | None,
    locale: str | None,
) -> dict[str, Any]:
    """Build the request fields shared by the map and crawl endpoints, omitting unset ones."""
    optional: dict[str, Any] = {
        "sitemap": sitemap,
        "domain_filter": domain_filter,
        "select_paths": select_paths,
        "exclude_paths": exclude_paths,
        "geo": geo,
        "locale": locale,
    }
    return {"url": url, **{key: value for key, value in optional.items() if value}}


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

_GEO_DESCRIPTION = 'Country to send the request from, as a full name, for example "United States".'
_LOCALE_DESCRIPTION = 'Interface language, for example "en-US".'


class _WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query.")
    geo: str | None = Field(default=None, description=_GEO_DESCRIPTION)
    locale: str | None = Field(default=None, description=_LOCALE_DESCRIPTION)
    page: int | None = Field(
        default=None, ge=1, description="Results page to return, 1 by default."
    )


class _SiteDiscoveryInput(BaseModel):
    url: str = Field(..., description="Absolute http or https URL of the site to start from.")
    sitemap: Literal["include", "only", "skip"] | None = Field(
        default=None,
        description="Whether to use the site's sitemap alongside its links, only the sitemap, "
        "or skip it.",
    )
    domain_filter: Literal["domain", "subdomain", "all"] | None = Field(
        default=None,
        description="Which hosts count as part of the site: the same domain, its subdomains "
        "too, or any host.",
    )
    select_paths: list[str] | None = Field(
        default=None,
        description="Regular expressions matched against URL paths. "
        "Only URLs matching one of them are kept.",
    )
    exclude_paths: list[str] | None = Field(
        default=None,
        description="Regular expressions matched against URL paths. "
        "URLs matching one of them are dropped.",
    )
    geo: str | None = Field(default=None, description=_GEO_DESCRIPTION)
    locale: str | None = Field(default=None, description=_LOCALE_DESCRIPTION)


class _MapInput(_SiteDiscoveryInput):
    limit: int = Field(
        default=_MAP_DEFAULT_LIMIT,
        ge=1,
        le=_MAP_LIMIT_MAX,
        description=f"Maximum number of URLs to return, up to {_MAP_LIMIT_MAX}. "
        f"Defaults to {_MAP_DEFAULT_LIMIT}.",
    )
    js_render: bool = Field(
        default=False,
        description="When true, renders pages with JavaScript to find links. "
        "Should only be on when asked by the user.",
    )


class _CrawlInput(_SiteDiscoveryInput):
    limit: int = Field(
        default=_CRAWL_DEFAULT_LIMIT,
        ge=1,
        le=_CRAWL_LIMIT_MAX,
        description=f"Maximum number of pages to crawl, up to {_CRAWL_LIMIT_MAX}. "
        f"Defaults to {_CRAWL_DEFAULT_LIMIT}.",
    )
    max_depth: int | None = Field(
        default=None,
        ge=1,
        le=_CRAWL_MAX_DEPTH_MAX,
        description="How many links deep to follow from the start URL.",
    )
    output: Literal["markdown", "html"] = Field(
        default="markdown", description="Format of the page content. Defaults to markdown."
    )


class _CrawlCancelInput(BaseModel):
    crawl_id: str = Field(
        ...,
        description="Id of the crawl to stop: status.crawl_id from a decodo_crawl result.",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class _DecodoDataApiTool(BaseTool):
    """Base for tools that call the Web Data API with an API key."""

    decodo_api_token: SecretStr = Field(
        default=SecretStr(""),
        description="Web Data API key. Falls back to the DECODO_API_TOKEN environment variable.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_api_token(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Load `DECODO_API_TOKEN` from the environment if not explicitly set.

        Args:
            values: Raw field values before model construction.

        Returns:
            Updated field values with the API key populated from env if needed.
        """
        return token_from_env(values)

    def _api(self) -> WebScrapingApi:
        """Build an SDK client for the configured API key.

        Returns:
            The SDK's Web Scraping API client, pointed at `data.decodo.com`.

        Raises:
            ValueError: If no API key is configured.
        """
        api_key = self.decodo_api_token.get_secret_value().strip()
        if not api_key:
            msg = (
                "A Decodo Web Data API key is required. Set it via the `decodo_api_token` "
                "field or the `DECODO_API_TOKEN` environment variable."
            )
            raise ValueError(msg)
        config = WebScrapingApiConfig(api_key=api_key, integration_header=INTEGRATION_HEADER)
        return DecodoClient(DecodoConfig(web_scraping_api=config)).web_scraping_api


class DecodoWebSearchTool(_DecodoDataApiTool):
    """Search the web and return ranked results with title, link and description.

    Unlike `DecodoSearchTool`, which scrapes a specific search engine's results
    page, this calls Decodo's search endpoint and returns a normalized result
    document. Needs a Web Data API key.

    Example:
        ```python
        from langchain_decodo import DecodoWebSearchTool

        tool = DecodoWebSearchTool()  # reads the API key from DECODO_API_TOKEN
        results = tool.invoke({"query": "LangChain release notes"})
        ```
    """

    name: str = "decodo_web_search"
    description: str = (
        "Search the web and retrieve ranked results with title, link and description. "
        "Returns a JSON object whose 'organic' list holds the results. "
        "To read a result, scrape its link with decodo_scrape_url."
    )
    args_schema: type[BaseModel] = _WebSearchInput

    def _run(
        self,
        query: str,
        geo: str | None = None,
        locale: str | None = None,
        page: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Run the search and return the result document as a JSON string.

        Args:
            query: Search query.
            geo: Country to send the request from.
            locale: Interface language.
            page: Results page to return.
            run_manager: Optional callback manager for tracing.

        Returns:
            JSON string of the search result document.

        Raises:
            ValueError: If no API key is configured.
            RuntimeError: On API errors, or when the search itself failed.
        """
        api = self._api()
        optional = {"geo": geo, "locale": locale, "page": page}
        params = {"query": query, **{k: v for k, v in optional.items() if v is not None}}
        response = _call_api(lambda: api.search(params))

        # A failed search comes back as HTTP 200 with `status: "failed"`.
        if response.get("status") == "failed":
            msg = f"Decodo search failed: {response.get('message') or 'no message given'}"
            raise RuntimeError(msg)

        return json.dumps(response, ensure_ascii=False, indent=2)


class DecodoMapTool(_DecodoDataApiTool):
    """List the URLs on a website without fetching their content.

    Needs a Web Data API key.

    Example:
        ```python
        from langchain_decodo import DecodoMapTool

        tool = DecodoMapTool()  # reads the API key from DECODO_API_TOKEN
        result = tool.invoke({"url": "https://www.python.org", "select_paths": ["^/downloads/"]})
        ```
    """

    name: str = "decodo_map"
    description: str = (
        "List the URLs on a website without fetching their content. "
        "Narrow large sites with select_paths or exclude_paths, then scrape only the pages "
        "you need with decodo_scrape_url."
    )
    args_schema: type[BaseModel] = _MapInput

    def _run(
        self,
        url: str,
        sitemap: str | None = None,
        domain_filter: str | None = None,
        select_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        geo: str | None = None,
        locale: str | None = None,
        limit: int = _MAP_DEFAULT_LIMIT,
        js_render: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Map the site and return the API response as a JSON string.

        Args:
            url: URL of the site to start from.
            sitemap: Whether to use the site's sitemap.
            domain_filter: Which hosts count as part of the site.
            select_paths: Path patterns a URL must match to be kept.
            exclude_paths: Path patterns that drop a URL.
            geo: Country to send the request from.
            locale: Interface language.
            limit: Maximum number of URLs to return.
            js_render: Whether to render pages with JavaScript to find links.
            run_manager: Optional callback manager for tracing.

        Returns:
            JSON string of the map response.

        Raises:
            ValueError: If no API key is configured.
            RuntimeError: On API errors or network failures, or when the target
                site could not be mapped.
        """
        api = self._api()
        params = _site_discovery_params(
            url, sitemap, domain_filter, select_paths, exclude_paths, geo, locale
        )
        params["limit"] = limit
        if js_render:
            params["JS_rendering"] = True
        response = _call_api(lambda: api.map(params))

        # Like a scrape, a failed map comes back as HTTP 200 with a failed status
        # inside `results`, or with no `results` at all.
        results: list[dict[str, Any]] = response.get("results") or []  # type: ignore[assignment]
        if not results or all(is_failed(entry) for entry in results):
            status = results[0].get("status_code") if results else response.get("status_code")
            message = response.get("message") or "no results returned"
            msg = f"Decodo map failed (status {status}): {message}"
            raise RuntimeError(msg)

        return json.dumps(response, ensure_ascii=False, indent=2)


class DecodoCrawlTool(_DecodoDataApiTool):
    """Crawl a website from a start URL and return the content of every page found.

    The tool runs the crawl with the SDK's `crawl()`, which polls until the crawl
    finishes, and returns the crawl's status with every page collected. A crawl
    that ends `faulted` or `cancelled`, or is still `pending` at `timeout`,
    returns what it collected so far instead of raising. `timeout` is also sent
    to the API as the crawl's `max_duration`, capped at the API's 4-hour limit,
    so Decodo stops a crawl the tool stopped waiting for. Needs a Web Data API key.

    Attributes:
        poll_interval: Seconds between status checks.
        timeout: Seconds to wait for the crawl before giving up.

    Example:
        ```python
        from langchain_decodo import DecodoCrawlTool

        tool = DecodoCrawlTool(timeout=300)  # reads the API key from DECODO_API_TOKEN
        result = tool.invoke({"url": "https://example.com", "limit": 5})
        ```
    """

    name: str = "decodo_crawl"
    description: str = (
        "Crawl a website from a start URL and retrieve the content of every page found. "
        "Waits until the crawl finishes. Check status.status in the result: anything other "
        'than "done" means the crawl stopped early and pages holds only what was collected. '
        '"pending" means the tool stopped waiting while the crawl still runs; stop it with '
        "decodo_crawl_cancel and status.crawl_id. status is null when it could not be read. "
        "Each page's own status says whether that page was fetched. "
        "To list a site without fetching content, use decodo_map instead."
    )
    args_schema: type[BaseModel] = _CrawlInput

    poll_interval: float = Field(default=_DEFAULT_POLL_INTERVAL, gt=0)
    timeout: float = Field(default=_DEFAULT_CRAWL_TIMEOUT, gt=0)

    def _run(
        self,
        url: str,
        sitemap: str | None = None,
        domain_filter: str | None = None,
        select_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        geo: str | None = None,
        locale: str | None = None,
        limit: int = _CRAWL_DEFAULT_LIMIT,
        max_depth: int | None = None,
        output: str = "markdown",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Crawl the site and return its final status and pages as a JSON string.

        Args:
            url: URL to start crawling from.
            sitemap: Whether to use the site's sitemap.
            domain_filter: Which hosts count as part of the site.
            select_paths: Path patterns a URL must match to be crawled.
            exclude_paths: Path patterns that skip a URL.
            geo: Country to send the requests from.
            locale: Interface language.
            limit: Maximum number of pages to crawl.
            max_depth: How many links deep to follow from the start URL.
            output: Format of the page content, `markdown` or `html`.
            run_manager: Optional callback manager for tracing.

        Returns:
            JSON string `{"status": {...} | null, "pages": [...]}`.

        Raises:
            ValueError: If no API key is configured.
            RuntimeError: On API errors or network failures.
        """
        api = self._api()
        params = _site_discovery_params(
            url, sitemap, domain_filter, select_paths, exclude_paths, geo, locale
        )
        params.update(limit=limit, output=output)
        if max_depth is not None:
            params["max_depth"] = max_depth
        params["max_duration"] = min(math.ceil(self.timeout), _MAX_DURATION_CEILING)

        def run_crawl() -> tuple[str | None, list[CrawlPage] | None]:
            try:
                pages = api.crawl(params, poll_interval=self.poll_interval, timeout=self.timeout)
            except (CrawlTimeoutError, CrawlError) as exc:
                return exc.crawl_id, None
            return _crawl_id_from_pages(pages), pages

        crawl_id, pages = _call_api(run_crawl)
        if pages is None:
            assert crawl_id is not None
            pages = _call_api(
                lambda: api.get_crawl_results(crawl_id), context=f"Crawl {crawl_id}: "
            )
        status = _final_status(api, crawl_id)
        return json.dumps({"status": status, "pages": pages}, ensure_ascii=False, indent=2)


def _crawl_id_from_pages(pages: list[CrawlPage]) -> str | None:
    """Read the crawl id from a page's `crawl_status` link; the SDK's `crawl()` drops it.

    The link has the form `https://data.decodo.com/v1/crawl/{crawl_id}/status`.
    """
    for page in pages:
        match = _CRAWL_STATUS_LINK.search(page.get("crawl_status") or "")
        if match:
            return match.group(1)
    return None


def _final_status(api: WebScrapingApi, crawl_id: str | None) -> CrawlStatusResponse | None:
    """Fetch the crawl's status, or `None` when there is no id or the request fails.

    The pages are already in hand at this point, so a failed status request
    should not turn the whole crawl into an error.
    """
    if crawl_id is None:
        return None
    try:
        return _call_api(lambda: api.get_crawl_status(crawl_id))
    except RuntimeError:
        return None


class DecodoCrawlCancelTool(_DecodoDataApiTool):
    """Stop a running crawl, for example one `DecodoCrawlTool` stopped waiting for.

    Needs a Web Data API key.

    Example:
        ```python
        import json

        from langchain_decodo import DecodoCrawlCancelTool, DecodoCrawlTool

        result = json.loads(DecodoCrawlTool(timeout=60).invoke({"url": "https://example.com"}))
        status = result["status"]
        if status and status["status"] == "pending":
            DecodoCrawlCancelTool().invoke({"crawl_id": status["crawl_id"]})
        ```
    """

    name: str = "decodo_crawl_cancel"
    description: str = (
        "Stop a crawl that decodo_crawl stopped waiting for, so it crawls no more pages. "
        "Use it with status.crawl_id from a decodo_crawl result whose status is pending."
    )
    args_schema: type[BaseModel] = _CrawlCancelInput

    def _run(
        self,
        crawl_id: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Cancel the crawl and return the API response as a JSON string.

        Args:
            crawl_id: Id of the crawl to stop.
            run_manager: Optional callback manager for tracing.

        Returns:
            JSON string of the cancel response.

        Raises:
            ValueError: If no API key is configured.
            RuntimeError: On API errors or network failures.
        """
        api = self._api()
        response = _call_api(lambda: api.cancel_crawl(crawl_id))
        return json.dumps(response, ensure_ascii=False, indent=2)
