"""LangChain tools for the Decodo web scraping API.

This module provides two tools:

- ``DecodoWebScrapeTool``: Scrapes any URL and returns the page content as
  markdown (or raw HTML/text).
- ``DecodoSearchTool``: Searches Google, Amazon, or Reddit and returns
  structured JSON results.

Both tools support two authentication modes selected via ``auth_mode``:

* ``"basic"`` *(default)* — username:password credentials encoded as a Basic
  auth token.  Uses the ``/v2/scrape`` endpoint.
* ``"token"`` — a plain API token.  Uses the ``/unified/v1/scrape`` endpoint.

In both modes the ``Authorization: Basic <value>`` header is used; only the
value and the target endpoint differ.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import httpx
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SecretStr, model_validator

_DEFAULT_BASE_URL = "https://scraper-api.decodo.com"

# Endpoint selected by auth_mode
_V2_SCRAPE_PATH = "/v2/scrape"
_UNIFIED_SCRAPE_PATH = "/unified/v1/scrape"

_DEFAULT_TIMEOUT = 180.0  # seconds

_ENGINE_TARGET_MAP: dict[str, str] = {
    "google": "google_search",
    "amazon": "amazon_search",
    "reddit": "google_search",  # Reddit searches are done via Google
}

_REDDIT_SITE_FILTER = "site:reddit.com"


def _scrape_path(auth_mode: str) -> str:
    """Return the API path for the given auth mode.

    Args:
        auth_mode: Either ``"basic"`` or ``"token"``.

    Returns:
        The URL path string for the scrape endpoint.
    """
    return _UNIFIED_SCRAPE_PATH if auth_mode == "token" else _V2_SCRAPE_PATH


def _build_headers(token: str) -> dict[str, str]:
    """Build HTTP headers for a Decodo API request.

    Args:
        token: Raw token string. For ``"basic"`` auth this is the
            base64-encoded ``username:password`` value; for ``"token"`` auth
            this is the plain API token.

    Returns:
        Dictionary of HTTP headers including ``Authorization``.
    """
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-integration": "langchain",
    }


def _do_scrape(
    token: str,
    base_url: str,
    payload: dict[str, Any],
    auth_mode: str = "basic",
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """POST to the appropriate Decodo scrape endpoint and return parsed JSON.

    Args:
        token: Raw token string (format depends on ``auth_mode``).
        base_url: Base URL of the Decodo API.
        payload: Request body to send as JSON.
        auth_mode: ``"basic"`` (username:password, ``/v2/scrape``) or
            ``"token"`` (API token, ``/unified/v1/scrape``).
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        RuntimeError: On timeout, network error, or non-2xx HTTP response.
    """
    url = f"{base_url}{_scrape_path(auth_mode)}"
    headers = _build_headers(token)

    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Decodo API request timed out after {timeout}s: {exc}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Decodo API network error: {exc}") from exc

    if not response.is_success:
        error_message: str
        try:
            body = response.json()
            error_message = body.get("message") or f"HTTP {response.status_code}"
        except Exception:
            error_message = f"HTTP {response.status_code}"
        raise RuntimeError(f"Decodo API error: {error_message}")

    return response.json()  # type: ignore[no-any-return]


def _is_failed(entry: dict[str, Any]) -> bool:
    """Return whether a result entry carries a failed (4xx/5xx or 6xx) status."""
    status = entry.get("status_code")
    return isinstance(status, int) and status >= 400


def _extract_content(response: dict[str, Any]) -> str:
    """Pull the first result's content string from a Decodo API response.

    Args:
        response: Parsed JSON response from the Decodo API.

    Returns:
        Content string. Returns an empty string if no results are present.
    """
    results = response.get("results", [])
    if not results:
        return ""
    first = results[0]
    content = first.get("content", "")
    if isinstance(content, str):
        return content
    # Some targets return structured content as a dict/list; serialise it.
    return json.dumps(content, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------


class _WebScrapeInput(BaseModel):
    url: str = Field(..., description="The full URL to scrape (must include scheme).")


class _SearchInput(BaseModel):
    query: str = Field(..., description="The search query string.")
    engine: str = Field(
        default="google",
        description=(
            "Search engine to use. Supported values: "
            "'google' (Google Search), 'amazon' (Amazon product search), "
            "'reddit' (Reddit posts via google_search). "
            "Defaults to 'google'."
        ),
    )
    num_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return (1-100).",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class DecodoWebScrapeTool(BaseTool):
    """Scrape any URL and return its content as markdown/text.

    The tool handles JavaScript rendering, anti-bot protection, and proxy
    rotation automatically via the Decodo API.

    Two authentication modes are supported via ``auth_mode``:

    * ``"basic"`` *(default)* — supply the base64-encoded
      ``username:password`` string as ``decodo_api_token`` (or set
      ``DECODO_API_TOKEN``).  Requests go to ``/v2/scrape``.
    * ``"token"`` — supply a plain API token as ``decodo_api_token`` (or set
      ``DECODO_API_TOKEN``).  Requests go to ``/unified/v1/scrape``.

    Both modes send ``Authorization: Basic <value>``; the value and endpoint
    differ.

    Attributes:
        decodo_api_token: Credential value. For ``"basic"`` mode this is the
            base64-encoded ``username:password``; for ``"token"`` mode this is
            the plain API token. Falls back to the ``DECODO_API_TOKEN``
            environment variable when not provided explicitly.
        auth_mode: Authentication mode — ``"basic"`` (default) or ``"token"``.
        base_url: Base URL of the Decodo Scraper API.

    Example — basic auth (default)::

        from langchain_decodo import DecodoWebScrapeTool

        tool = DecodoWebScrapeTool(decodo_api_token="base64(user:pass)")
        result = tool.run("https://example.com")

    Example — token auth::

        tool = DecodoWebScrapeTool(
            decodo_api_token="your-api-token",
            auth_mode="token",
        )
        result = tool.run("https://example.com")
    """

    name: str = "decodo_scrape_url"
    description: str = (
        "Scrape the full content of any web page given its URL. "
        "Returns the page content as markdown or plain text. "
        "Handles JavaScript-rendered pages, CAPTCHAs, and geo-blocked content "
        "automatically. Use this when you need the complete text of a specific URL. "
        "Input: a valid URL string (must include http:// or https://)."
    )
    args_schema: type[BaseModel] = _WebScrapeInput

    decodo_api_token: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "Credential value. For 'basic' auth: base64-encoded username:password. "
            "For 'token' auth: plain API token. "
            "Falls back to the DECODO_API_TOKEN environment variable."
        ),
    )
    auth_mode: Literal["basic", "token"] = Field(
        default="basic",
        description=(
            "Authentication mode. "
            "'basic' (default): username:password credentials, uses /v2/scrape. "
            "'token': plain API token, uses /unified/v1/scrape."
        ),
    )
    base_url: str = Field(
        default=_DEFAULT_BASE_URL,
        description="Base URL of the Decodo Scraper API.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_api_token(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Load ``DECODO_API_TOKEN`` from the environment if not explicitly set.

        Args:
            values: Raw field values before model construction.

        Returns:
            Updated field values with the token populated from env if needed.
        """
        if not values.get("decodo_api_token"):
            env_token = os.environ.get("DECODO_API_TOKEN", "")
            values["decodo_api_token"] = SecretStr(env_token)
        return values

    def _run(
        self,
        url: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Scrape the given URL and return its content.

        Args:
            url: The full URL to scrape (must include scheme).
            run_manager: Optional callback manager for tracing.

        Returns:
            Page content as a string (markdown or plain text).

        Raises:
            ValueError: If no API token is configured.
            RuntimeError: On API errors or network failures.
        """
        token = self.decodo_api_token.get_secret_value()
        if not token:
            raise ValueError(
                "Decodo API token is required. Set it via the ``decodo_api_token`` "
                "field or the ``DECODO_API_TOKEN`` environment variable."
            )

        payload: dict[str, Any] = {
            "target": "universal",
            "url": url,
            "markdown": True,
        }
        response = _do_scrape(token, self.base_url, payload, auth_mode=self.auth_mode)
        content = _extract_content(response)
        return content if content else "(No content returned by Decodo API)"


class DecodoSearchTool(BaseTool):
    """Search Google, Amazon, or Reddit and return structured JSON results.

    The tool maps the ``engine`` parameter to the appropriate Decodo target:

    * ``google``  → ``google_search`` target
    * ``amazon``  → ``amazon_search`` target
    * ``reddit``  → ``google_search`` with ``site:reddit.com`` prepended

    Two authentication modes are supported via ``auth_mode``:

    * ``"basic"`` *(default)* — supply the base64-encoded
      ``username:password`` string as ``decodo_api_token`` (or set
      ``DECODO_API_TOKEN``).  Requests go to ``/v2/scrape``.
    * ``"token"`` — supply a plain API token as ``decodo_api_token`` (or set
      ``DECODO_API_TOKEN``).  Requests go to ``/unified/v1/scrape``.

    Attributes:
        decodo_api_token: Credential value. Falls back to ``DECODO_API_TOKEN``.
        auth_mode: Authentication mode — ``"basic"`` (default) or ``"token"``.
        engine: Default search engine (``google``, ``amazon``, or ``reddit``).

    Example — basic auth (default)::

        from langchain_decodo import DecodoSearchTool

        tool = DecodoSearchTool(decodo_api_token="base64(user:pass)")
        results = tool.run({"query": "best Python web scraping libraries"})

    Example — token auth::

        tool = DecodoSearchTool(
            decodo_api_token="your-api-token",
            auth_mode="token",
        )
        results = tool.run({"query": "best Python web scraping libraries"})
    """

    name: str = "decodo_search"
    description: str = (
        "Search the web using Google, Amazon, or Reddit and return structured results. "
        "Returns a JSON list of search results, each with 'content', 'url', and "
        "'status_code' fields. "
        "Input must be a JSON object with: "
        "'query' (required, the search string), "
        "'engine' (optional: 'google', 'amazon', or 'reddit'; default 'google'), "
        "'num_results' (optional: integer 1-100; default 10). "
        "Use 'amazon' to search for products, 'reddit' for community discussions."
    )
    args_schema: type[BaseModel] = _SearchInput

    decodo_api_token: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "Credential value. For 'basic' auth: base64-encoded username:password. "
            "For 'token' auth: plain API token. "
            "Falls back to the DECODO_API_TOKEN environment variable."
        ),
    )
    auth_mode: Literal["basic", "token"] = Field(
        default="basic",
        description=(
            "Authentication mode. "
            "'basic' (default): username:password credentials, uses /v2/scrape. "
            "'token': plain API token, uses /unified/v1/scrape."
        ),
    )
    engine: str = Field(
        default="google",
        description="Default search engine: 'google', 'amazon', or 'reddit'.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_api_token(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Load ``DECODO_API_TOKEN`` from the environment if not explicitly set.

        Args:
            values: Raw field values before model construction.

        Returns:
            Updated field values with the token populated from env if needed.
        """
        if not values.get("decodo_api_token"):
            env_token = os.environ.get("DECODO_API_TOKEN", "")
            values["decodo_api_token"] = SecretStr(env_token)
        return values

    def _run(
        self,
        query: str,
        engine: str = "google",
        num_results: int = 10,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Execute a search and return results as a JSON string.

        Args:
            query: The search query string.
            engine: Search engine to use (``google``, ``amazon``, ``reddit``).
            num_results: Maximum number of results to return (1-100).
            run_manager: Optional callback manager for tracing.

        Returns:
            JSON string — a list of result objects with ``content``,
            ``status_code``, and ``url`` fields.

        Raises:
            ValueError: If no API token is configured.
            RuntimeError: On API errors or network failures.
        """
        token = self.decodo_api_token.get_secret_value()
        if not token:
            raise ValueError(
                "Decodo API token is required. Set it via the ``decodo_api_token`` "
                "field or the ``DECODO_API_TOKEN`` environment variable."
            )

        engine = engine.lower().strip()
        target = _ENGINE_TARGET_MAP.get(engine, "google_search")

        # Prepend Reddit site filter when using the reddit pseudo-engine.
        effective_query = f"{_REDDIT_SITE_FILTER} {query}" if engine == "reddit" else query

        payload: dict[str, Any] = {
            "target": target,
            "query": effective_query,
            "limit": num_results,
            "parse": True,
        }

        response = _do_scrape(token, _DEFAULT_BASE_URL, payload, auth_mode=self.auth_mode)
        results = response.get("results", [])

        # A failed scrape comes back as HTTP 200, either with a failed status
        # inside `results` or with no `results` at all (e.g. status 613). Raise
        # so agents can retry instead of reading it as "no results".
        if not results or all(_is_failed(entry) for entry in results):
            status = results[0].get("status_code") if results else response.get("status_code")
            message = response.get("message") or "no results returned"
            raise RuntimeError(f"Decodo search failed (status {status}): {message}")

        serialisable = []
        for entry in results:
            content = entry.get("content", "")
            serialisable.append(
                {
                    "content": content,
                    "status_code": entry.get("status_code"),
                    "url": entry.get("url", ""),
                }
            )

        return json.dumps(serialisable, ensure_ascii=False, indent=2)
