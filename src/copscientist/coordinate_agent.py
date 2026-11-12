"""Orchestrate a research round over the current tree frontier."""

from __future__ import annotations

import uuid

from .coding_agent import CodingAgent
from .config import Config
from .evolution_agent import EvolutionAgent
from .experiment_agent import ExperimentAgent
from .flow_graph import FlowGraph
from .ideation_agent import IdeationAgent
from .utils import Candidate, Evaluation


class CoordinateAgent:
    def __init__(self, number: int, config: Config, ideation: IdeationAgent,
                 coding: CodingAgent, experiment: ExperimentAgent,
                 evolution: EvolutionAgent, tree: FlowGraph):
        self.number = number
        self.config = config
        self.ideation, self.coding, self.experiment = ideation, coding, experiment
        self.evolution, self.tree = evolution, tree

    def research_round(self, number: int) -> list[Candidate]:
        parent = self.tree.select(self.config.max_depth, self.number + number)
        if parent is None:
            return []
        self.tree.expanded.add(parent.id)
        memories = self.evolution.retrieve(self.config.task.name)
        ideas = self.ideation.propose(parent, memories, self.config.ideas_per_round, self.tree)
        produced = []
        for idea in ideas:
            identifier = uuid.uuid4().hex[:12]
            directory = self.config.run_path / "candidates" / identifier
            filename = "gpt.py" if self.config.task.kind == "heuristic" else "candidate.py"
            path = directory / filename
            evaluation = Evaluation(status="failed", feedback="Code generation did not complete")
            for attempt in range(self.config.debug_attempts + 1):
                try:
                    self.coding.implement(idea, parent, path, evaluation.feedback if attempt else "")
                    evaluation = self.experiment.evaluate(path)
                except (ValueError, OSError) as exc:
                    evaluation = Evaluation(status="failed", feedback=str(exc))
                if evaluation.status == "ok":
                    break
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Generation failed\n", encoding="utf-8")
            reflection = self.experiment.summarize(idea, evaluation, parent.evaluation.score)
            candidate = Candidate(id=identifier, round=number, depth=parent.depth + 1,
                                  parent_id=parent.id, idea=idea, code_path=str(path),
                                  evaluation=evaluation, reflection=reflection)
            self.tree.add(candidate)
            self.evolution.record_candidate(self.config.task.name, candidate)
            self.evolution.distill(self.config.task.name, candidate, parent,
                                   self.config.task.objective)
            produced.append(candidate)
        return produced
