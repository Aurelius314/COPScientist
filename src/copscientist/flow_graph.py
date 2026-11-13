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

    def add(self, solution: Solution, *, depth: int | None = None) -> None:
        if solution.id in self.nodes:
            raise ValueError(f"Duplicate solution: {solution.id}")
        if depth is None and solution.parent_id and solution.parent_id not in self.nodes:
            raise ValueError(f"Missing parent: {solution.parent_id}")
        self.nodes[solution.id] = solution
        self.children[solution.id] = []
        self.depths[solution.id] = solution.depth if depth is None else depth
        if solution.parent_id in self.children:
            self.children[solution.parent_id].append(solution.id)

    def mark_done(self, solution_id: str) -> None:
        self.done.add(solution_id)

    def select(self, max_depth: int | None, rank: int = 0) -> Solution | None:
        frontier = [solution for solution in self.nodes.values()
                    if solution.id not in self.done and not self.children[solution.id]
                    and (max_depth is None or self.depths[solution.id] < max_depth)
                    and solution.evaluation.status == "ok"]
        frontier.sort(key=lambda solution: solution.evaluation.score or 0,
                      reverse=self.objective == "max")
        return frontier[rank % len(frontier)] if frontier else None

    def leaves(self) -> list[Solution]:
        return [solution for solution in self.nodes.values()
                if not self.children[solution.id]]

    def finished(self, max_depth: int | None) -> bool:
        return self.select(max_depth) is None

    def to_dict(self) -> dict:
        return {"nodes": [
            {"id": solution.id, "parent_id": solution.parent_id,
             "depth": self.depths[solution.id], "children": self.children[solution.id],
             "done": solution.id in self.done,
             "status": solution.evaluation.status, "score": solution.evaluation.score}
            for solution in self.nodes.values()
        ]}

    def best(self) -> Solution | None:
        best = None
        for solution in self.nodes.values():
            if solution.evaluation.status == "ok" and (
                best is None or better(solution.evaluation.score, best.evaluation.score, self.objective)
            ):
                best = solution
        return best
