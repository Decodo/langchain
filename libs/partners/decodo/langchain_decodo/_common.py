"""Helpers shared by the scrape tools, the document loader and the Web Data API tools."""

from __future__ import annotations

import os
from typing import Any

from pydantic import SecretStr

INTEGRATION_HEADER = "langchain-python"

TOKEN_ENV_VAR = "DECODO_API_TOKEN"


def token_from_env(values: dict[str, Any]) -> dict[str, Any]:
    """Fill `decodo_api_token` from `DECODO_API_TOKEN` when it was not given.

    Args:
        values: Raw field values before model construction.

    Returns:
        The same values, with `decodo_api_token` populated from env if needed.
    """
    if not values.get("decodo_api_token"):
        values["decodo_api_token"] = SecretStr(os.environ.get(TOKEN_ENV_VAR, ""))
    return values


def is_failed(entry: dict[str, Any]) -> bool:
    """Return whether a result entry carries a failed (4xx/5xx or 6xx) status."""
    status = entry.get("status_code")
    return isinstance(status, int) and status >= 400
