"""Organize solutions as a research tree, following OR-Agent's FlowGraph."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from .utils import Evaluation, Idea, Solution, better


@dataclass(eq=False)
class Node:
    """A solution and its position in one research tree."""

    solution: Solution | None = None
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)
    depth: int = 0
    is_done: bool = False


class FlowGraph:
    """Solution tree. A round has one root; the global store may have several."""

    def __init__(self, objective: str | None = None, root: Solution | None = None,
                 *, root_solution: Solution | None = None,
                 checkpoint: str | Path | None = None,
                 project_root: Path | None = None):
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.objective = objective
        self.obj_type = objective
        self._root: Node | None = None
        self._nodes: list[Node] = []
        # Keep the existing COPScientist lookup interfaces for core and prompts.
        self.nodes: dict[str | int, Solution] = {}
        self.children: dict[str | int, list[str | int]] = {}
        self.depths: dict[str | int, int] = {}
        if checkpoint is not None:
            self.load(checkpoint)
        else:
            if objective not in {"min", "max"}:
                raise ValueError(f"Invalid objective type: {objective}")
            initial = root_solution if root_solution is not None else root
            if initial is not None:
                self.add(initial, depth=0)

    @property
    def root(self) -> Node | None:
        return self._root

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def done(self) -> set[str | int]:
        return {node.solution.id for node in self._nodes if node.is_done}

    def _append_node(self, node: Node, parent: Node | None = None,
                     depth: int | None = None) -> None:
        if node.solution is None:
            raise ValueError("A research node must contain a solution")
        solution_id = node.solution.id
        if solution_id in self.nodes:
            raise ValueError(f"Duplicate solution: {solution_id}")
        if parent is not None and parent not in self._nodes:
            raise ValueError("Parent node is not in this flow graph")
        if node.parent is not None and node.parent is not parent:
            raise ValueError("Node already belongs to another parent")
        node.parent = parent
        node.depth = (parent.depth + 1 if parent is not None else
                      (node.solution.depth if depth is None else depth))
        if parent is not None:
            parent.children.append(node)
            self.children[parent.solution.id].append(solution_id)
        elif self._root is None:
            self._root = node
        self._nodes.append(node)
        self.nodes[solution_id] = node.solution
        self.children[solution_id] = []
        self.depths[solution_id] = node.depth

    def add(self, parent_node: Node | Solution,
            children_nodes: Node | list[Node] | None = None,
            *, depth: int | None = None) -> None:
        """Append child node(s), as in OR-Agent; also accept a Solution directly."""
        if isinstance(parent_node, Solution):
            if children_nodes is not None:
                raise TypeError("A Solution cannot be paired with children_nodes")
            solution = parent_node
            parent = None
            if solution.parent_id is not None:
                parent = next((node for node in self._nodes
                               if node.solution.id == solution.parent_id), None)
                if parent is None and depth is None:
                    raise ValueError(f"Missing parent: {solution.parent_id}")
            self._append_node(Node(solution=solution), parent, depth)
            return
        if children_nodes is None:
            raise TypeError("children_nodes is required when adding to a Node")
        children = children_nodes if isinstance(children_nodes, list) else [children_nodes]
        for child in children:
            self._append_node(child, parent_node)

    def get_best_undone_leaf(self, max_depth: int | None = None) -> Node | None:
        """Return the best evaluated leaf that can still be expanded."""
        best_leaf = None
        for leaf in self._nodes:
            if (leaf.children or leaf.is_done or
                    (max_depth is not None and leaf.depth >= max_depth) or
                    leaf.solution.evaluation.status != "ok" or
                    leaf.solution.evaluation.score is None):
                continue
            if best_leaf is None or better(leaf.solution.evaluation.score,
                                           best_leaf.solution.evaluation.score,
                                           self.obj_type):
                best_leaf = leaf
        return best_leaf

    def get_leaves(self) -> list[Node]:
        return [node for node in self._nodes if not node.children]

    def get_done_leaves(self) -> list[Node]:
        return [node for node in self.get_leaves() if node.is_done]

    def check_research_finished(self) -> bool:
        return all(node.is_done for node in self.get_leaves())

    def mark_done(self, solution_id: str | int) -> None:
        if solution_id not in self.nodes:
            raise KeyError(solution_id)
        node = next(node for node in self._nodes if node.solution.id == solution_id)
        node.is_done = True

    def select(self, max_depth: int | None, rank: int = 0) -> Solution | None:
        """Compatibility interface used by CoordinateAgent."""
        frontier = [node for node in self._nodes
                    if not node.children and not node.is_done
                    and (max_depth is None or node.depth < max_depth)
                    and node.solution.evaluation.status == "ok"
                    and node.solution.evaluation.score is not None]
        frontier.sort(key=lambda node: node.solution.evaluation.score,
                      reverse=self.obj_type == "max")
        return frontier[rank % len(frontier)].solution if frontier else None

    def leaves(self) -> list[Solution]:
        return [node.solution for node in self.get_leaves()]

    def finished(self, max_depth: int | None) -> bool:
        return self.select(max_depth) is None

    def best(self) -> Solution | None:
        best_solution = None
        for node in self._nodes:
            solution = node.solution
            if solution.evaluation.status == "ok" and (
                    best_solution is None or
                    better(solution.evaluation.score, best_solution.evaluation.score,
                           self.obj_type)):
                best_solution = solution
        return best_solution

    def to_dict(self) -> dict:
        """Compact context for prompts and round detail files."""
        return {"nodes": [
            {"id": node.solution.id,
             "parent_id": node.parent.solution.id if node.parent else node.solution.parent_id,
             "depth": node.depth,
             "children": [child.solution.id for child in node.children],
             "done": node.is_done,
             "status": node.solution.evaluation.status,
             "score": node.solution.evaluation.score}
            for node in self._nodes
        ]}

    def _checkpoint_path(self, checkpoint: str | Path) -> Path:
        path = Path(checkpoint)
        if path.is_absolute():
            return path if path.suffix.lower() == ".json" else path / "flow_graph.json"
        if path.name != str(path) or path.name in {".", ".."}:
            raise ValueError("Checkpoint must be a directory name or absolute path")
        return self.project_root / "checkpoints" / path / "flow_graph.json"

    def save(self, checkpoint: str | Path) -> None:
        """Save the tree in OR-Agent's indexed-node checkpoint layout."""
        path = self._checkpoint_path(checkpoint)
        path.parent.mkdir(parents=True, exist_ok=True)
        indices = {node: index for index, node in enumerate(self._nodes)}
        payload = {
            "objective": self.obj_type,
            "root_index": indices[self._root] if self._root is not None else None,
            "nodes": [
                {"solution": node.solution.to_dict(), "is_done": node.is_done,
                 "depth": node.depth,
                 "children_indices": [indices[child] for child in node.children]}
                for node in self._nodes
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=4),
                        encoding="utf-8")

    def load(self, checkpoint: str | Path) -> None:
        """Restore solutions first, then their parent and child references."""
        payload = json.loads(self._checkpoint_path(checkpoint).read_text(encoding="utf-8"))
        objective = payload.get("objective", self.objective)
        if objective not in {"min", "max"}:
            raise ValueError(f"Invalid objective type: {objective}")
        self.objective = self.obj_type = objective
        self._root = None
        self._nodes = []
        self.nodes = {}
        self.children = {}
        self.depths = {}
        for item in payload["nodes"]:
            data = dict(item["solution"])
            if isinstance(data.get("idea"), dict):
                data["idea"] = Idea(**data["idea"])
            if isinstance(data.get("evaluation"), dict):
                data["evaluation"] = Evaluation(**data["evaluation"])
            solution = Solution(**data)
            node = Node(solution=solution, depth=item.get("depth", solution.depth),
                        is_done=item["is_done"])
            if solution.id in self.nodes:
                raise ValueError(f"Duplicate solution: {solution.id}")
            self._nodes.append(node)
            self.nodes[solution.id] = solution
            self.depths[solution.id] = node.depth
            self.children[solution.id] = []
        for index, item in enumerate(payload["nodes"]):
            parent = self._nodes[index]
            for child_index in item["children_indices"]:
                child = self._nodes[child_index]
                parent.children.append(child)
                child.parent = parent
                self.children[parent.solution.id].append(child.solution.id)
        root_index = payload.get("root_index")
        if root_index is not None:
            self._root = self._nodes[root_index]

    def visualize(self, show_details: bool = False,
                  file: TextIO = sys.stdout) -> None:
        print(self.visualize_str(show_details), file=file)

    def visualize_str(self, show_details: bool = False) -> str:
        """Render OR-Agent's expanded, pending and terminal tree markers."""
        def render(node: Node, prefix: str = "", is_last: bool = True) -> list[str]:
            is_root = node.parent is None
            connector = "" if is_root else ("    └── " if is_last else "    ├── ")
            score = node.solution.evaluation.score
            label = (f"{node.solution.id} ({score:.2f})" if score is not None
                     else f"{node.solution.id} (N/A)")
            if show_details and node.solution.idea is not None:
                idea = node.solution.idea
                label += f': "{idea.title if isinstance(idea, Idea) else idea}"'
            status = "✓" if node.is_done else ("○" if not node.children else "⊕")
            lines = [f"{prefix}{connector}{status} Node {label}"]
            child_prefix = prefix + ("" if is_root else
                                     ("        " if is_last else "    │   "))
            for index, child in enumerate(node.children):
                lines.extend(render(child, child_prefix,
                                    index == len(node.children) - 1))
            return lines
        lines = ["Format: Node <ID> (<score>): \"<idea>\"",
                 "Legend:",
                 "  ⊕ = solution expanded with improved children solution(s)",
                 "  ○ = solution pending expansion",
                 "  ✓ = terminal solution (local optimum, no improvement found)",
                 "=" * 40]
        for node in self._nodes:
            if node.parent is None:
                lines.extend(render(node))
        lines.extend(["=" * 40,
                      f"Total nodes: {len(self)} | Total expanded solutions: "
                      f"{len(self) - len(self.get_leaves())} | Total pending leaves: "
                      f"{len(self.get_leaves()) - len(self.get_done_leaves())} | "
                      f"Total terminal leaves: {len(self.get_done_leaves())}"])
        return "\n".join(lines)


if __name__ == "__main__":
    # For debugging purposes, without creating files or calling an evaluator.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    seed = Solution(idea=Idea("Seed", "", "", ""),
                    evaluation=Evaluation("ok", score=10.0))
    graph = FlowGraph("min", root_solution=seed)
    child = Solution(idea=Idea("Improvement", "", "", ""),
                     parent_id=seed.id, evaluation=Evaluation("ok", score=8.0))
    graph.add(graph.root, Node(solution=child))
    graph.visualize(show_details=True)
    print(f"Best unfinished leaf: {graph.get_best_undone_leaf().solution.id}")
