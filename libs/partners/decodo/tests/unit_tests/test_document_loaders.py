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
