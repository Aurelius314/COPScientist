"""Cross-domain scholarly and web retrieval with source provenance."""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
from html.parser import HTMLParser
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .utils import Source


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict:
    request = Request(url, headers={"User-Agent": "COPScientist/0.1 (research)", **(headers or {})})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def _post_json(url: str, payload: dict) -> dict:
    request = Request(url, data=json.dumps(payload).encode("utf-8"),
                      headers={"User-Agent": "COPScientist/0.1 (research)",
                               "Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=20) as response:
        return json.load(response)


class _PageText(HTMLParser):
    """Keep readable webpage text without adding a crawler dependency."""

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden += 1
        elif not self.hidden and tag in {"p", "br", "li", "h1", "h2", "h3", "section", "article"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.hidden:
            self.hidden -= 1
        elif not self.hidden and tag in {"p", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def _get_page(url: str, max_chars: int = 8000) -> str:
    if not url.startswith(("https://", "http://")):
        raise ValueError("Only HTTP(S) source pages can be read")
    request = Request(url, headers={"User-Agent": "COPScientist/0.1 (research)"})
    with urlopen(request, timeout=10) as response:
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type and "text/plain" not in content_type:
            raise ValueError(f"Unsupported page content type: {content_type}")
        raw = response.read(200_000)
        charset = response.headers.get_content_charset() or "utf-8"
    decoded = raw.decode(charset, errors="replace")
    if "html" in content_type:
        parser = _PageText()
        parser.feed(decoded)
        decoded = "".join(parser.parts)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", decoded)).strip()[:max_chars]


class LiteratureSearch:
    def __init__(self, fetch=_get_json, *, post=_post_json, fetch_page=_get_page):
        self.fetch = fetch
        self.post = post
        self.fetch_page = fetch_page

    def papers(self, query: str, limit: int) -> list[Source]:
        params = urlencode({"query.bibliographic": query, "rows": limit,
                            "select": "DOI,title,abstract,published,URL"})
        data = self.fetch(f"https://api.crossref.org/works?{params}")
        result = []
        for item in data.get("message", {}).get("items", []):
            title = item.get("title", [])
            doi = item.get("DOI")
            if not title or not doi:
                continue
            dates = item.get("published", {}).get("date-parts", [[]])
            year = dates[0][0] if dates and dates[0] else None
            abstract = re.sub(r"<[^>]+>", " ", item.get("abstract") or "")
            result.append(Source(title=title[0], url=f"https://doi.org/{doi}",
                                 kind="paper", query=query, year=year,
                                 abstract=" ".join(abstract.split())[:1500],
                                 content_level="abstract" if abstract else "metadata"))
        return result

    def web(self, query: str, limit: int) -> list[Source]:
        key = os.environ.get("TAVILY_API_KEY")
        if not key:
            return []
        data = self.post("https://api.tavily.com/search",
                         {"api_key": key, "query": query, "max_results": min(limit, 20),
                          "search_depth": "basic", "include_answer": False})
        results = data.get("results", [])
        return [Source(title=item.get("title") or "", url=item["url"], kind="web",
                       query=query, abstract=item.get("content") or item.get("description") or "",
                       content_level="snippet")
                for item in results if item.get("url")]

    def read_sources(self, sources: list[Source], max_pages: int = 6,
                     max_chars: int = 8000) -> list[str]:
        """Read a bounded set of pages, retaining metadata when a page is unavailable."""
        errors: list[str] = []
        # Prefer web pages: Crossref often yields an abstract, while a web hit is only a snippet.
        selected = sorted(sources, key=lambda source: source.kind != "web")[:max_pages]
        if not selected:
            return errors
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(selected))) as executor:
            futures = [executor.submit(self.fetch_page, source.url, max_chars=max_chars)
                       for source in selected]
            for source, future in zip(selected, futures):
                try:
                    content = future.result()
                    if not content:
                        raise ValueError("Page has no readable text")
                    source.content = content[:max_chars]
                    source.content_level = "page"
                except Exception as exc:
                    errors.append(f"Page read for {source.url} failed: {exc}")
        return errors

    def review(self, literature_queries: list[str], web_queries: list[str],
               limit: int = 3, required: bool = False) -> tuple[list[Source], list[str]]:
        sources: dict[str, Source] = {}
        errors: list[str] = []
        for kind, queries in (("paper", literature_queries), ("web", web_queries)):
            for query in queries:
                try:
                    found = self.papers(query, limit) if kind == "paper" else self.web(query, limit)
                    if kind == "web" and not found and not os.environ.get("TAVILY_API_KEY"):
                        errors.append("Web search skipped: set TAVILY_API_KEY")
                    for source in found:
                        sources.setdefault(source.url.lower(), source)
                except Exception as exc:
                    errors.append(f"{kind} search for {query!r} failed: {exc}")
        if required and (errors or not sources):
            raise RuntimeError("Search required but incomplete: " + "; ".join(errors))
        return list(sources.values()), errors
