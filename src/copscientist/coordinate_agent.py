"""Orchestrate a research round over the current tree frontier."""

from __future__ import annotations

import json
import uuid

from .coding_agent import CodingAgent
from .config import Config
from .evolution_agent import EvolutionAgent
from .experiment_agent import ExperimentAgent
from .flow_graph import FlowGraph
from .ideation_agent import IdeationAgent
from .utils import (Solution, Model, better, experiment_allocation, load_prompt,
                    parse_json, render_prompt, solution_code_path)


class CoordinateAgent:
    def __init__(self, number: int, config: Config, ideation: IdeationAgent,
                 coding: CodingAgent, experiment: ExperimentAgent,
                 evolution: EvolutionAgent, tree: FlowGraph,
                 model: Model | None = None):
        self.number = number
        self.config = config
        self.ideation, self.coding, self.experiment = ideation, coding, experiment
        self.evolution, self.tree = evolution, tree
        self.model = model or ideation.model
        self.flow_graph: FlowGraph | None = None
        knowledge = config.problem_path / "external_knowledge.txt"
        self.external_knowledge = (knowledge.read_text(encoding="utf-8")
                                   if knowledge.is_file() else "")
        self.long_term_reflection = self.external_knowledge
        self.experiment_summaries: list[str] = []

    def state_dict(self) -> dict:
        return {"long_term_reflection": self.long_term_reflection,
                "experiment_summaries": self.experiment_summaries}

    def load_state(self, state: dict) -> None:
        self.long_term_reflection = state.get("long_term_reflection", self.external_knowledge)
        self.experiment_summaries = list(state.get("experiment_summaries", []))

    def update_long_term_reflection(self, solution: Solution | None = None,
                                    *, force: bool = False) -> None:
        """Accumulate findings, then compress at OR-Agent's configured cadence."""
        if solution is not None and solution.reflection:
            self.experiment_summaries.append(
                f"{solution.idea.title}: score={solution.evaluation.score}; {solution.reflection}")
        if not self.experiment_summaries:
            return
        period = self.config.reflection_period
        if not force and (period <= 0 or len(self.experiment_summaries) < period):
            return
        context = {"previous_reflection": self.long_term_reflection,
                   "experiment_summaries": self.experiment_summaries,
                   "word_limit": self.config.reflection_compression}
        try:
            response = parse_json(self.model.complete(
                load_prompt("system_long_term_reflection_copscientist.txt"),
                render_prompt("user_long_term_reflection_copscientist.txt",
                              json.dumps(context, ensure_ascii=False)),
            ))
            if isinstance(response.get("reflection"), str) and response["reflection"].strip():
                reflection = response["reflection"]
                if self.config.reflection_compression is not None:
                    reflection = " ".join(reflection.split()[:self.config.reflection_compression])
                self.long_term_reflection = reflection
        except Exception:
            self.long_term_reflection = (self.long_term_reflection + "\n" +
                                         "\n".join(self.experiment_summaries))[-6000:]
        self.experiment_summaries.clear()

    def research_round(self, number: int, root: Solution) -> list[Solution]:
        """OR-Agent-style round: expand best unfinished leaves until exhausted."""
        self.flow_graph = FlowGraph(self.config.task.objective, root=root)
        if self.config.verbose:
            print(f">>>[CoordinateAgent {self.number}] Round {number} root: {root.id} "
                  f"(score={root.evaluation.score})")
        solution_count = 0
        while parent := self.flow_graph.select(self.config.max_depth):
            local_depth = self.flow_graph.depths[parent.id]
            attempts = 5 if local_depth == 0 else 1
            for _ in range(attempts):
                memories = (self.evolution.retrieve(self.config.task.name)
                            if self.config.evolution_enabled else [])
                ideas = self.ideation.run(parent, memories, self.config.ideas_per_round,
                                          self.flow_graph, self.long_term_reflection)
                for idea in ideas:
                    solution_count += 1
                    identifier = uuid.uuid4().hex[:12]
                    solution = Solution(id=identifier, round=number, depth=parent.depth + 1,
                                         parent_id=parent.id, idea=idea,
                                         coordinate_agent_id=self.number,
                                         solution_count=solution_count)
                    path = solution_code_path(self.config.run_path, self.config.task.kind,
                                              solution.id_str())
                    evaluation = self.coding.run(idea, parent, path,
                                                 self.long_term_reflection)
                    if not path.exists():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text("# Generation failed\n", encoding="utf-8")
                    if local_depth == 0 and evaluation.status != "ok":
                        # OR-Agent filters invalid root children before experiments.
                        continue
                    best = self.tree.best()
                    is_elitist = (evaluation.score is not None and best is not None
                                  and best.evaluation.score is not None
                                  and abs(evaluation.score - best.evaluation.score) < 1e-4)
                    repeats = self.config.experiment_repeats
                    if self.config.dynamic_experiment_allocation:
                        repeats = experiment_allocation(
                            parent.evaluation.score, evaluation.score, self.config.task.objective,
                            repeats, self.config.min_experiment_repeats,
                            self.config.max_experiment_repeats_cap,
                            self.config.improvement_threshold,
                            self.config.elitist_experiment_factor, is_elitist)
                    elif is_elitist:
                        repeats = int(repeats * self.config.elitist_experiment_factor)
                    evaluation, history, reflection, compact_summaries = self.experiment.run(
                        idea, path, evaluation, parent.evaluation.score,
                        self.long_term_reflection, max_repeats=repeats, parent=parent)
                    # OR-Agent gives an invalid post-experiment solution one final code repair.
                    if evaluation.score is None:
                        evaluation = self.coding.debug(
                            idea, parent, path, evaluation, self.long_term_reflection)
                    solution.code_path = str(path)
                    solution.code = path.read_text(encoding="utf-8")
                    solution.evaluation = evaluation
                    solution.reflection = reflection
                    solution.intermediate = history
                    solution.compact_summaries = compact_summaries
                    self.tree.add(solution)
                    if self.config.verbose:
                        print(f">>>[CoordinateAgent {self.number}] Solution {solution.id}: "
                              f"status={evaluation.status}, score={evaluation.score}")
                    self.evolution.record_solution(self.config.task.name, solution)
                    if self.config.evolution_enabled:
                        self.evolution.distill(self.config.task.name, solution, parent,
                                               self.config.task.objective)
                    if evaluation.status == "ok":
                        self.update_long_term_reflection(solution, force=True)
                    # Root children may temporarily worsen; deeper children must improve.
                    if evaluation.status == "ok" and (
                        local_depth == 0 or better(evaluation.score, parent.evaluation.score,
                                                    self.config.task.objective)
                    ):
                        self.flow_graph.add(solution, depth=local_depth + 1)
                if self.flow_graph.children[parent.id]:
                    break
            if local_depth == 0 and not self.flow_graph.children[parent.id]:
                raise RuntimeError("Coordinate Agent could not produce a valid root child in five attempts")
            self.flow_graph.mark_done(parent.id)
        if self.config.reflection_clearance_period and number % self.config.reflection_clearance_period == 0:
            self.long_term_reflection = self.external_knowledge
            self.experiment_summaries.clear()
        details = self.config.run_path / "details"
        details.mkdir(parents=True, exist_ok=True)
        (details / f"coordinate_{self.number}_round_{number}_tree.json").write_text(
            json.dumps(self.flow_graph.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return [leaf for leaf in self.flow_graph.leaves() if leaf.id != root.id]
