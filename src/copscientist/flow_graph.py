"""Research tree: implementations are children of the ideas they refine."""

from __future__ import annotations

from .utils import Candidate, better


class FlowGraph:
    def __init__(self, objective: str):
        self.objective = objective
        self.nodes: dict[str, Candidate] = {}
        self.expanded: set[str] = set()

    def add(self, candidate: Candidate) -> None:
        if candidate.id in self.nodes:
            raise ValueError(f"Duplicate candidate: {candidate.id}")
        if candidate.parent_id and candidate.parent_id not in self.nodes:
            raise ValueError(f"Missing parent: {candidate.parent_id}")
        self.nodes[candidate.id] = candidate

    def select(self, max_depth: int, rank: int = 0) -> Candidate | None:
        frontier = [candidate for candidate in self.nodes.values()
                    if candidate.id not in self.expanded and candidate.depth < max_depth
                    and candidate.evaluation.status == "ok"]
        frontier.sort(key=lambda candidate: candidate.evaluation.score or 0,
                      reverse=self.objective == "max")
        return frontier[rank % len(frontier)] if frontier else None

    def best(self) -> Candidate | None:
        best = None
        for candidate in self.nodes.values():
            if candidate.evaluation.status == "ok" and (
                best is None or better(candidate.evaluation.score, best.evaluation.score, self.objective)
            ):
                best = candidate
        return best
