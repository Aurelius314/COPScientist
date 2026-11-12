"""COPScientist entry point."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from .coding_agent import CodingAgent
from .config import Config
from .coordinate_agent import CoordinateAgent
from .evolution_agent import EvolutionAgent
from .experiment_agent import ExperimentAgent
from .flow_graph import FlowGraph
from .ideation_agent import IdeationAgent
from .search import LiteratureSearch
from .utils import Candidate, ChatModel, Idea, Model


class COPScientist:
    def __init__(self, config: Config, model: Model | None = None,
                 search: LiteratureSearch | None = None):
        self.config = config
        self.model = model or ChatModel(config.model, config.base_url, config.api_key_env)
        self.search = search or LiteratureSearch(config.brave_key_env)
        self.tree = FlowGraph(config.task.objective)

    def run(self) -> Candidate | None:
        config = self.config
        config.problem_path  # validate before creating outputs
        config.run_path.mkdir(parents=True, exist_ok=True)
        (config.run_path / "resolved_config.json").write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        evolution = EvolutionAgent(config.run_path / "memory.sqlite3")
        try:
            experiment = ExperimentAgent(config, self.model)
            ideation = IdeationAgent(config, self.model, self.search)
            coding = CodingAgent(config, self.model)
            sources, errors = ideation.prepare_review()
            evolution.record_sources(sources)
            (config.run_path / "literature_review.json").write_text(
                json.dumps({"review": ideation.review, "sources": [asdict(s) for s in sources],
                            "search_errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
            seed = config.task.seed_file
            if not seed:
                raise ValueError("task.seed_file is required to start research")
            seed_path = (config.problem_path / seed).resolve()
            if not seed_path.is_relative_to(config.problem_path) or not seed_path.is_file():
                raise FileNotFoundError(seed_path)
            baseline_id = uuid.uuid4().hex[:12]
            base_dir = config.run_path / "candidates" / baseline_id
            base_dir.mkdir(parents=True, exist_ok=True)
            baseline_path = base_dir / ("gpt.py" if config.task.kind == "heuristic" else "candidate.py")
            shutil.copyfile(seed_path, baseline_path)
            baseline = Candidate(
                id=baseline_id, round=0, depth=0, parent_id=None,
                idea=Idea(title="Baseline", rationale="Provided seed implementation",
                          algorithm="Existing implementation", expected_effect="Reference score"),
                code_path=str(baseline_path), evaluation=experiment.evaluate(baseline_path),
            )
            self.tree.add(baseline)
            evolution.record_candidate(config.task.name, baseline)
            evolution.distill(config.task.name, baseline, None, config.task.objective)
            if baseline.evaluation.status != "ok":
                raise RuntimeError(f"Baseline failed: {baseline.evaluation.feedback}")
            coordinates = [CoordinateAgent(i, config, ideation, coding, experiment,
                                           evolution, self.tree)
                           for i in range(config.coordinate_agents)]
            for round_number in range(1, config.rounds + 1):
                for agent in coordinates:
                    agent.research_round(round_number)
            best = self.tree.best()
            validation = experiment.evaluate(Path(best.code_path), validation=True) if best else None
            (config.run_path / "result.json").write_text(json.dumps({
                "task": config.task.name, "best": best.to_dict() if best else None,
                "validation": asdict(validation) if validation else None,
                "candidates": [node.to_dict() for node in self.tree.nodes.values()],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            return best
        finally:
            evolution.close()
