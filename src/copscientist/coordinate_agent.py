"""Orchestrate a research round over the current tree frontier."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict

from .coding_agent import CodingAgent
from .config import Config
from .evolution_agent import EvolutionAgent
from .experiment_agent import ExperimentAgent
from .flow_graph import FlowGraph
from .ideation_agent import IdeationAgent
from .utils import Candidate, better, load_prompt, parse_json, render_prompt


class CoordinateAgent:
    def __init__(self, number: int, config: Config, ideation: IdeationAgent,
                 coding: CodingAgent, experiment: ExperimentAgent,
                 evolution: EvolutionAgent, tree: FlowGraph):
        self.number = number
        self.config = config
        self.ideation, self.coding, self.experiment = ideation, coding, experiment
        self.evolution, self.tree = evolution, tree
        self.flow_graph: FlowGraph | None = None
        self.long_term_reflection = ""

    def update_long_term_reflection(self, candidate: Candidate) -> None:
        """Condense an experiment finding for later branches and rounds."""
        context = {"previous_reflection": self.long_term_reflection,
                   "idea": asdict(candidate.idea), "score": candidate.evaluation.score,
                   "experiment_summary": candidate.reflection}
        try:
            response = parse_json(self.ideation.model.complete(
                load_prompt("system_long_term_reflection_copscientist.txt"),
                render_prompt("user_long_term_reflection_copscientist.txt",
                              json.dumps(context, ensure_ascii=False)),
            ))
            if isinstance(response.get("reflection"), str):
                self.long_term_reflection = response["reflection"][:6000]
        except Exception:
            self.long_term_reflection = (self.long_term_reflection + "\n" +
                                         candidate.reflection)[-6000:]

    def research_round(self, number: int, root: Candidate) -> list[Candidate]:
        """OR-Agent-style round: expand best unfinished leaves until exhausted."""
        self.flow_graph = FlowGraph(self.config.task.objective, root=root)
        produced: list[Candidate] = []
        while parent := self.flow_graph.select(self.config.max_depth):
            local_depth = self.flow_graph.depths[parent.id]
            memories = self.evolution.retrieve(self.config.task.name)
            ideas = self.ideation.propose(parent, memories, self.config.ideas_per_round,
                                          self.flow_graph, self.long_term_reflection)
            for idea in ideas:
                identifier = uuid.uuid4().hex[:12]
                directory = self.config.run_path / "candidates" / identifier
                filename = "gpt.py" if self.config.task.kind == "heuristic" else "candidate.py"
                path = directory / filename
                evaluation = self.coding.run(idea, parent, path,
                                             self.long_term_reflection)
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("# Generation failed\n", encoding="utf-8")
                evaluation, history, reflection = self.experiment.run(
                    idea, path, evaluation, parent.evaluation.score,
                    self.long_term_reflection)
                candidate = Candidate(id=identifier, round=number, depth=parent.depth + 1,
                                      parent_id=parent.id, idea=idea, code_path=str(path),
                                      evaluation=evaluation, reflection=reflection,
                                      intermediate=history)
                self.tree.add(candidate)
                self.evolution.record_candidate(self.config.task.name, candidate)
                self.evolution.distill(self.config.task.name, candidate, parent,
                                       self.config.task.objective)
                produced.append(candidate)
                if evaluation.status == "ok":
                    self.update_long_term_reflection(candidate)
                # Like OR-Agent, the root may start a promising branch after a
                # temporary downgrade. Deeper branches require improvement.
                if evaluation.status == "ok" and (
                    local_depth == 0 or better(evaluation.score, parent.evaluation.score,
                                                self.config.task.objective)
                ):
                    self.flow_graph.add(candidate, depth=local_depth + 1)
            self.flow_graph.mark_done(parent.id)
        details = self.config.run_path / "details"
        details.mkdir(parents=True, exist_ok=True)
        (details / f"coordinate_{self.number}_round_{number}_tree.json").write_text(
            json.dumps(self.flow_graph.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return produced
