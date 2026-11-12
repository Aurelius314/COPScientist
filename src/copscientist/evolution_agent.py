"""Distill tested ideas and experiments into reusable research memory."""

from __future__ import annotations

from pathlib import Path

from .research_database import ResearchDatabase
from .utils import Candidate, Source, better


class EvolutionAgent:
    def __init__(self, path: Path, database: ResearchDatabase | None = None):
        self.database = database or ResearchDatabase(path)

    def close(self) -> None:
        self.database.close()

    def record_sources(self, sources: list[Source]) -> None:
        self.database.record_sources(sources)

    def record_candidate(self, task: str, candidate: Candidate) -> None:
        self.database.record_candidate(task, candidate)

    def retrieve(self, task: str, limit: int = 8) -> list[dict]:
        return self.database.retrieve(task, limit)

    def distill(self, task: str, candidate: Candidate, parent: Candidate | None,
                objective: str) -> None:
        """Store an evidence-linked idea outcome and experiment lesson."""
        evaluation = candidate.evaluation
        comparison = (
            "improved" if parent and better(evaluation.score, parent.evaluation.score, objective)
            else "did not improve" if parent and evaluation.score is not None
            else "initial result" if evaluation.score is not None else "failed"
        )
        ideation = f"{candidate.idea.title}: {candidate.idea.rationale} Outcome: {comparison}."
        experiment = (
            f"{candidate.idea.title}: status={evaluation.status}; score={evaluation.score}; "
            f"reflection={candidate.reflection[:1200]}; feedback={evaluation.feedback[:600]}"
        )
        evidence = {"candidate_id": candidate.id, "parent_id": candidate.parent_id,
                    "metrics": evaluation.metrics, "sources": candidate.idea.sources}
        self.database.record_memories(task, candidate.id, ideation, experiment, evidence)
