"""Cross-domain scholarly and web retrieval with source provenance."""

from __future__ import annotations

import json
import os
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .utils import Source


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict:
    request = Request(url, headers={"User-Agent": "COPScientist/0.1 (research)", **(headers or {})})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


class LiteratureSearch:
    def __init__(self, brave_key_env: str = "BRAVE_SEARCH_API_KEY", fetch=_get_json):
        self.brave_key_env = brave_key_env
        self.fetch = fetch

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
            abstract = re.sub(r"<[^>]+>", " ", item.get("abstract", ""))
            result.append(Source(title=title[0], url=f"https://doi.org/{doi}",
                                 kind="paper", query=query, year=year,
                                 abstract=" ".join(abstract.split())[:1500]))
        return result

    def web(self, query: str, limit: int) -> list[Source]:
        key = os.environ.get(self.brave_key_env)
        if not key:
            return []
        params = urlencode({"q": query, "count": min(limit, 20)})
        data = self.fetch(f"https://api.search.brave.com/res/v1/web/search?{params}",
                          {"X-Subscription-Token": key})
        return [Source(title=item.get("title", ""), url=item["url"], kind="web",
                       query=query, abstract=item.get("description", ""))
                for item in data.get("web", {}).get("results", []) if item.get("url")]

    def review(self, literature_queries: list[str], web_queries: list[str],
               limit: int = 3, required: bool = False) -> tuple[list[Source], list[str]]:
        sources: dict[str, Source] = {}
        errors: list[str] = []
        for kind, queries in (("paper", literature_queries), ("web", web_queries)):
            for query in queries:
                try:
                    found = self.papers(query, limit) if kind == "paper" else self.web(query, limit)
                    if kind == "web" and not found and not os.environ.get(self.brave_key_env):
                        errors.append(f"Web search skipped: set {self.brave_key_env}")
                    for source in found:
                        sources.setdefault(source.url.lower(), source)
                except Exception as exc:
                    errors.append(f"{kind} search for {query!r} failed: {exc}")
        if required and (errors or not sources):
            raise RuntimeError("Search required but incomplete: " + "; ".join(errors))
        return list(sources.values()), errors
