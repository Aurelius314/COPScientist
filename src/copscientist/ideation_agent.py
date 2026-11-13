"""
Ideation Agent: Source-grounded idea generation with web search for heuristic and solver design.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .flow_graph import FlowGraph
from .search import LiteratureSearch
from .utils import Candidate, Idea, Model, Source, load_prompt, parse_json, render_prompt


def _grounded(items: object, field: str, allowed: set[str]) -> list[dict]:
    if not isinstance(items, list):
        return []
    return [
        {field: item[field], "source_urls": item["source_urls"]}
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get(field), str)
        and isinstance(item.get("source_urls"), list)
        and item["source_urls"]
        and all(url in allowed for url in item["source_urls"])
    ]


class IdeationAgent:
    def __init__(self, config: Config, model: Model, search: LiteratureSearch):
        self.config, self.model, self.search = config, model, search
        self.sources: list[Source] = []
        self.review: dict = {}

    def prepare_review(self) -> tuple[list[Source], list[str]]:
        self.sources, errors = self.search.review(
            [f"{self.config.task.name} combinatorial optimization algorithm",
             *self.config.literature_queries],
            [f"{self.config.task.name} heuristic solver implementation",
             *self.config.web_queries],
            self.config.search_results, self.config.search_required,
        )
        payload = [{**asdict(source), "abstract": source.abstract[:500]}
                   for source in self.sources[:20]]
        if payload:
            raw_review = parse_json(self.model.complete(
                load_prompt("system_literature_review_copscientist.txt"),
                render_prompt("user_literature_review_copscientist.txt",
                              json.dumps(payload, ensure_ascii=False)),
            ))
            allowed = {source.url for source in self.sources}
            self.review = {
                "claims": _grounded(raw_review.get("claims"), "claim", allowed),
                "transferable_methods": _grounded(raw_review.get("transferable_methods"), "method", allowed),
                "limitations": raw_review.get("limitations", []),
            }
        else:
            self.review = {"claims": [], "transferable_methods": [], "limitations": errors}
        return self.sources, errors

    def propose(self, parent: Candidate | None, memory: list[dict], count: int,
                tree: FlowGraph, long_term_reflection: str = "") -> list[Idea]:
        parent_context = None if parent is None else {
            "idea": asdict(parent.idea), "score": parent.evaluation.score,
            "feedback": parent.evaluation.feedback[-1500:],
            "code": Path(parent.code_path).read_text(encoding="utf-8")[-5000:],
        }
        context = {
            "task": asdict(self.config.task), "parent": parent_context,
            "long_term_reflection": long_term_reflection,
            "review": self.review, "sources": [asdict(source) for source in self.sources],
            "memory": memory[:6], "tree_progress": [
                {"id": node.id, "parent": node.parent_id, "title": node.idea.title,
                 "score": node.evaluation.score, "status": node.evaluation.status}
                for node in list(tree.nodes.values())[-15:]
            ], "count": count,
        }
        allowed = {source.url for source in self.sources}
        user_prompt = render_prompt("user_idea_generation_copscientist.txt",
                                    json.dumps(context, ensure_ascii=False))
        for _ in range(5):
            try:
                result = parse_json(self.model.complete(
                    load_prompt("system_idea_generation_copscientist.txt"), user_prompt))
                ideas = [Idea(title=item["title"], rationale=item["rationale"],
                              algorithm=item["algorithm"], expected_effect=item["expected_effect"],
                              sources=[url for url in item.get("sources", []) if url in allowed])
                         for item in result.get("ideas", [])[:count]]
                if ideas:
                    return ideas
            except (ValueError, KeyError, TypeError):
                pass
        raise RuntimeError("Ideation Agent failed to return valid ideas after five attempts")
