# langchain-decodo

LangChain partner package for the [Decodo](https://decodo.com) web scraping API.

Provides **LangChain tools** for scraping, web search, site mapping and crawling,
plus a **document loader**, that let LLM agents and RAG pipelines fetch live web
content without managing proxies, JavaScript rendering, or anti-bot protection.

## Installation

```bash
pip install langchain-decodo
```

## Authentication

Use the Web Data API key from your Web Data API subscription on the
[Decodo Dashboard](https://dashboard.decodo.com/web-data/playground). All classes read it from the `DECODO_API_TOKEN` environment variable, or you can pass it explicitly. `DecodoWebScrapeTool`, `DecodoSearchTool` and `DecodoLoader` also need `auth_mode="token"` when using the key:

```bash
export DECODO_API_TOKEN="your-decodo-api-key"
```

```python
from langchain_decodo import DecodoWebScrapeTool

tool = DecodoWebScrapeTool(auth_mode="token")
```

Older plans only have a basic authentication token (base64-encoded
`username:password`). That is the default `auth_mode="basic"`, so omit
`auth_mode` and export the basic token as `DECODO_API_TOKEN`.

`DecodoWebSearchTool`, `DecodoMapTool`, `DecodoCrawlTool` and
`DecodoCrawlCancelTool` call endpoints that only accept an API key, so they need
no `auth_mode` and do not work with a basic authentication token.

## Components

### `DecodoWebScrapeTool`

Scrape any URL and return its full content as markdown or plain text.
Handles JavaScript rendering, CAPTCHAs, and geo-blocking automatically.

```python
from langchain_decodo import DecodoWebScrapeTool

tool = DecodoWebScrapeTool(auth_mode="token")  # reads DECODO_API_TOKEN from env
content = tool.run("https://example.com")
print(content)
```

Pass explicitly:

```python
from pydantic import SecretStr
from langchain_decodo import DecodoWebScrapeTool

tool = DecodoWebScrapeTool(
    decodo_api_token=SecretStr("YOUR_API_KEY"),
    auth_mode="token",
)
```

### `DecodoSearchTool`

Search Google, Amazon, or Reddit and return structured JSON results.

```python
from langchain_decodo import DecodoSearchTool

tool = DecodoSearchTool(auth_mode="token")

# Google search (default)
results = tool.run({"query": "LangChain latest release", "engine": "google"})

# Amazon product search
results = tool.run({"query": "Python programming book", "engine": "amazon"})

# Reddit discussion search
results = tool.run({"query": "best web scraping libraries", "engine": "reddit"})
```

Returns a JSON string — a list of objects with `content`, `url`, and
`status_code` fields.

Supported engines:

| `engine` | Decodo target                       | Description           |
| -------- | ----------------------------------- | --------------------- |
| `google` | `google_search`                     | Google SERP           |
| `amazon` | `amazon_search`                     | Amazon product search |
| `reddit` | `google_search` + `site:reddit.com` | Reddit via Google     |

### `DecodoWebSearchTool`

Search the web and return ranked results with title, link and description.
Unlike `DecodoSearchTool`, which scrapes a specific engine's results page, this
returns one normalized result document. Requires an API key.

```python
from langchain_decodo import DecodoWebSearchTool

tool = DecodoWebSearchTool()
results = tool.invoke({"query": "coffee shops", "geo": "United States", "page": 1})
```

Returns a JSON string; the results are under `organic`, alongside
`top_stories`, `people_also_ask` and `related_searches` when the search has them.

### `DecodoMapTool`

List the URLs on a website without fetching their content. Requires an API key.
Raises an error when the site could not be mapped, for example when it blocks
the request.

```python
from langchain_decodo import DecodoMapTool

tool = DecodoMapTool()
result = tool.invoke(
    {
        "url": "https://www.python.org",
        "select_paths": ["^/downloads/"],
        "limit": 50,
    }
)
```

Optional arguments: `sitemap` (`include`, `only` or `skip`), `domain_filter`
(`domain`, `subdomain` or `all`), `select_paths` and `exclude_paths` (regular
expressions matched against URL paths), `geo`, `locale`, `limit` (up to 500,
default 100) and `js_render`.

### `DecodoCrawlTool` and `DecodoCrawlCancelTool`

Crawl a website from a start URL and return the content of every page found.
The tool runs the crawl through the Decodo SDK, waits for it to finish and
returns `{"status": {...}, "pages": [...]}`. Requires an API key.

```python
from langchain_decodo import DecodoCrawlTool

tool = DecodoCrawlTool(timeout=300, poll_interval=5)
result = tool.invoke({"url": "https://example.com", "limit": 10, "max_depth": 2})
```

Takes the same optional arguments as `DecodoMapTool` except `js_render`, plus
`limit` (up to 100, default 10), `max_depth` and `output` (`markdown` or
`html`).

`status` is the crawl's status response and each entry in `pages` is a page as
the API returns it, including its own `status`. A `status.status` other than
`done` means the crawl stopped early and `pages` holds only what was collected.
`status` is `null` when it could not be read, for example when the crawl
collected no pages.

`timeout` (seconds, default 600) is also sent as the crawl's `max_duration`,
capped at the API's 4-hour limit. A crawl still running at `timeout` comes back
as `pending` with the pages collected so far; Decodo stops it at the same limit
shortly after. To stop it sooner, pass `status.crawl_id` to
`DecodoCrawlCancelTool`:

```python
import json

from langchain_decodo import DecodoCrawlCancelTool, DecodoCrawlTool

result = json.loads(DecodoCrawlTool(timeout=60).invoke({"url": "https://example.com"}))
status = result["status"]
if status and status["status"] == "pending":
    DecodoCrawlCancelTool().invoke({"crawl_id": status["crawl_id"]})
```

### `DecodoLoader`

Load one or more URLs as LangChain `Document` objects for use in RAG pipelines.

```python
from langchain_decodo import DecodoLoader

loader = DecodoLoader(
    urls=[
        "https://python.org/about/",
        "https://docs.python.org/3/whatsnew/3.12.html",
    ],
    auth_mode="token",
)
docs = loader.load()

for doc in docs:
    print(doc.metadata["url"], "—", len(doc.page_content), "chars")
```

Each `Document` has:

- `page_content` — scraped text/markdown.
- `metadata["url"]` — the source URL.
- `metadata["source"]` — same as `url` (LangChain convention).
- `metadata["status_code"]` — HTTP status from the target site.

## LangChain agent example

```bash
pip install langchain langchain-openai langchain-decodo
```

```python
from langchain.agents import create_agent
from langchain_decodo import DecodoMapTool, DecodoWebScrapeTool, DecodoWebSearchTool

agent = create_agent(
    model="openai:gpt-6.1-sol",
    tools=[
        DecodoWebSearchTool(),
        DecodoMapTool(),
        DecodoWebScrapeTool(auth_mode="token"),
    ],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the latest stable version of Python? Check python.org.",
            }
        ]
    }
)
print(result["messages"][-1].text)
```

## RAG pipeline example

```bash
pip install langchain-openai langchain-text-splitters langchain-decodo numpy
```

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_decodo import DecodoLoader

docs = DecodoLoader(urls=["https://python.org/about/"], auth_mode="token").load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

store = InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings())
context = "\n\n".join(
    doc.page_content for doc in store.similarity_search("What is Python used for?", k=4)
)

llm = ChatOpenAI(model="gpt-6.1-sol")
answer = llm.invoke(
    f"Answer using only this context:\n\n{context}\n\nQuestion: What is Python used for?"
)
print(answer.text)
```

## Links

- [Decodo website](https://decodo.com)
- [Decodo API documentation](https://help.decodo.com)
- [Decodo Dashboard](https://dashboard.decodo.com/)
- [LangChain documentation](https://docs.langchain.com/oss/python/langchain/overview)
