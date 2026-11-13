"""Research tree: implementations are children of the ideas they refine."""

from __future__ import annotations

from .utils import Solution, better


class FlowGraph:
    def __init__(self, objective: str, root: Solution | None = None):
        self.objective = objective
        self.nodes: dict[str, Solution] = {}
        self.children: dict[str, list[str]] = {}
        self.depths: dict[str, int] = {}
        self.done: set[str] = set()
        if root is not None:
            self.add(root, depth=0)

    def add(self, candidate: Solution, *, depth: int | None = None) -> None:
        if candidate.id in self.nodes:
            raise ValueError(f"Duplicate solution: {candidate.id}")
        if depth is None and candidate.parent_id and candidate.parent_id not in self.nodes:
            raise ValueError(f"Missing parent: {candidate.parent_id}")
        self.nodes[candidate.id] = candidate
        self.children[candidate.id] = []
        self.depths[candidate.id] = candidate.depth if depth is None else depth
        if candidate.parent_id in self.children:
            self.children[candidate.parent_id].append(candidate.id)

    def mark_done(self, candidate_id: str) -> None:
        self.done.add(candidate_id)

    def select(self, max_depth: int | None, rank: int = 0) -> Solution | None:
        frontier = [candidate for candidate in self.nodes.values()
                    if candidate.id not in self.done and not self.children[candidate.id]
                    and (max_depth is None or self.depths[candidate.id] < max_depth)
                    and candidate.evaluation.status == "ok"]
        frontier.sort(key=lambda candidate: candidate.evaluation.score or 0,
                      reverse=self.objective == "max")
        return frontier[rank % len(frontier)] if frontier else None

    def leaves(self) -> list[Solution]:
        return [candidate for candidate in self.nodes.values()
                if not self.children[candidate.id]]

    def finished(self, max_depth: int | None) -> bool:
        return self.select(max_depth) is None

    def to_dict(self) -> dict:
        return {"nodes": [
            {"id": candidate.id, "parent_id": candidate.parent_id,
             "depth": self.depths[candidate.id], "children": self.children[candidate.id],
             "done": candidate.id in self.done,
             "status": candidate.evaluation.status, "score": candidate.evaluation.score}
            for candidate in self.nodes.values()
        ]}

    def best(self) -> Solution | None:
        best = None
        for candidate in self.nodes.values():
            if candidate.evaluation.status == "ok" and (
                best is None or better(candidate.evaluation.score, best.evaluation.score, self.objective)
            ):
                best = candidate
        return best
