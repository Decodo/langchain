# langchain-decodo

LangChain partner package for the [Decodo](https://decodo.com) web scraping API.

Provides two **LangChain tools** and a **document loader** that let LLM agents
and RAG pipelines fetch live web content without managing proxies, JavaScript
rendering, or anti-bot protection.

## Installation

```bash
pip install langchain-decodo
```

## Authentication

All classes read your Decodo API token from the `DECODO_API_TOKEN` environment
variable, or you can pass it explicitly.

```bash
export DECODO_API_TOKEN="your-decodo-api-token"
```

Get a token from the [Decodo Dashboard](https://app.decodo.com).

## Components

### `DecodoWebScrapeTool`

Scrape any URL and return its full content as markdown or plain text.
Handles JavaScript rendering, CAPTCHAs, and geo-blocking automatically.

```python
from langchain_decodo import DecodoWebScrapeTool

tool = DecodoWebScrapeTool()  # reads DECODO_API_TOKEN from env
content = tool.run("https://example.com")
print(content)
```

Pass explicitly:

```python
from pydantic import SecretStr
from langchain_decodo import DecodoWebScrapeTool

tool = DecodoWebScrapeTool(decodo_api_token=SecretStr("YOUR_TOKEN"))
```

### `DecodoSearchTool`

Search Google, Amazon, or Reddit and return structured JSON results.

```python
from langchain_decodo import DecodoSearchTool

tool = DecodoSearchTool()

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

| `engine` | Decodo target | Description |
|---|---|---|
| `google` | `google_search` | Google SERP |
| `amazon` | `amazon_search` | Amazon product search |
| `reddit` | `google_search` + `site:reddit.com` | Reddit via Google |

### `DecodoLoader`

Load one or more URLs as LangChain `Document` objects for use in RAG pipelines.

```python
from langchain_decodo import DecodoLoader

loader = DecodoLoader(
    urls=[
        "https://python.org/about/",
        "https://docs.python.org/3/whatsnew/3.12.html",
    ],
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
from langchain_decodo import DecodoSearchTool, DecodoWebScrapeTool

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[DecodoWebScrapeTool(), DecodoSearchTool()],
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
print(result["messages"][-1].content)
```

## RAG pipeline example

```bash
pip install langchain-openai langchain-text-splitters langchain-decodo
```

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_decodo import DecodoLoader

docs = DecodoLoader(urls=["https://python.org/about/"]).load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

store = InMemoryVectorStore.from_documents(chunks, OpenAIEmbeddings())
context = "\n\n".join(
    doc.page_content for doc in store.similarity_search("What is Python used for?", k=4)
)

llm = ChatOpenAI(model="gpt-4o-mini")
answer = llm.invoke(
    f"Answer using only this context:\n\n{context}\n\nQuestion: What is Python used for?"
)
print(answer.content)
```

## Links

- [Decodo website](https://decodo.com)
- [Decodo API documentation](https://help.decodo.com)
- [Decodo Dashboard](https://app.decodo.com)
- [LangChain documentation](https://python.langchain.com)
