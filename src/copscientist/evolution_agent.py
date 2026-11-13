"""Distill tested ideas and experiments into reusable research memory."""

from __future__ import annotations

from pathlib import Path

from .research_database import ResearchDatabase
from .utils import Solution, Source, better


class EvolutionAgent:
    def __init__(self, path: Path, database: ResearchDatabase | None = None):
        self.database = database or ResearchDatabase(path)

    def close(self) -> None:
        self.database.close()

    def record_sources(self, sources: list[Source]) -> None:
        self.database.record_sources(sources)

    def record_solution(self, task: str, solution: Solution) -> None:
        self.database.record_solution(task, solution)

    def retrieve(self, task: str, limit: int = 8) -> list[dict]:
        return self.database.retrieve(task, limit)

    def distill(self, task: str, solution: Solution, parent: Solution | None,
                objective: str) -> None:
        """Store an evidence-linked idea outcome and experiment lesson."""
        evaluation = solution.evaluation
        comparison = (
            "improved" if parent and better(evaluation.score, parent.evaluation.score, objective)
            else "did not improve" if parent and evaluation.score is not None
            else "initial result" if evaluation.score is not None else "failed"
        )
        ideation = f"{solution.idea.title}: {solution.idea.rationale} Outcome: {comparison}."
        experiment = (
            f"{solution.idea.title}: status={evaluation.status}; score={evaluation.score}; "
            f"reflection={solution.reflection[:1200]}; feedback={evaluation.feedback[:600]}"
        )
        evidence = {"solution_id": solution.id, "parent_id": solution.parent_id,
                    "metrics": evaluation.metrics, "sources": solution.idea.sources}
        self.database.record_memories(task, solution.id, ideation, experiment, evidence)
