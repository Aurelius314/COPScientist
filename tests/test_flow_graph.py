"""Flow graph checkpoint and frontier behavior."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from copscientist.flow_graph import FlowGraph, Node
from copscientist.utils import Evaluation, Idea, Solution


def solution(identifier: str, score: float, parent_id: str | None = None) -> Solution:
    return Solution(id=identifier, parent_id=parent_id,
                    idea=Idea(identifier, "", "", ""),
                    evaluation=Evaluation("ok", score=score))


def test_node_tree_frontier_and_checkpoint():
    root = solution("root", 10)
    graph = FlowGraph("min", root_solution=root)
    first = Node(solution=solution("first", 8, "root"))
    second = Node(solution=solution("second", 9, "root"))
    graph.add(graph.root, [first, second])

    assert first.parent is graph.root
    assert first.depth == 1
    assert graph.get_best_undone_leaf() is first
    assert graph.select(max_depth=1) is None

    graph.mark_done("first")
    assert graph.get_best_undone_leaf() is second
    assert "✓ Node first (8.00)" in graph.visualize_str(show_details=True)

    run_dir = Path(__file__).resolve().parents[1] / "runs"
    run_dir.mkdir(exist_ok=True)
    temporary = run_dir / f"test_flow_graph_{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        checkpoint = temporary / "round"
        graph.save(checkpoint)
        restored = FlowGraph(checkpoint=checkpoint)
        assert restored.root.solution.id == "root"
        assert len(restored) == 3
        assert restored.root.children[0].parent is restored.root
        assert restored.root.children[0].depth == 1
        assert restored.get_best_undone_leaf().solution.id == "second"
        assert restored.to_dict() == graph.to_dict()

        restored.mark_done("second")
        assert restored.check_research_finished()
    finally:
        shutil.rmtree(temporary)
