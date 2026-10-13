"""Unit tests for DecodoLoader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_decodo import DecodoLoader


@patch("langchain_decodo.document_loaders.httpx.post")
def test_loader_requests_markdown(mock_post: MagicMock) -> None:
    response = MagicMock()
    response.is_success = True
    response.json.return_value = {"results": [{"content": "# Title", "status_code": 200}]}
    mock_post.return_value = response

    docs = DecodoLoader(urls="https://example.com", api_token="tok").load()

    assert mock_post.call_args[1]["json"]["markdown"] is True
    assert docs[0].page_content == "# Title"


@patch("langchain_decodo.document_loaders.httpx.post")
def test_loader_basic_mode_uses_scraper_api(mock_post: MagicMock) -> None:
    response = MagicMock()
    response.is_success = True
    response.json.return_value = {"results": [{"content": "x", "status_code": 200}]}
    mock_post.return_value = response

    DecodoLoader(urls="https://example.com", api_token="tok").load()

    assert mock_post.call_args[0][0] == "https://scraper-api.decodo.com/v2/scrape"
    assert mock_post.call_args[1]["headers"]["Authorization"] == "Basic tok"


@patch("langchain_decodo.document_loaders.httpx.post")
def test_loader_token_mode_uses_data_api(mock_post: MagicMock) -> None:
    response = MagicMock()
    response.is_success = True
    response.json.return_value = {"results": [{"content": "x", "status_code": 200}]}
    mock_post.return_value = response

    DecodoLoader(urls="https://example.com", api_token="key", auth_mode="token").load()

    assert mock_post.call_args[0][0] == "https://data.decodo.com/v1/scrape"
    assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer key"
