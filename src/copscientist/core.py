"""COPScientist entry point and OR-Agent-style round checkpoints."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .coding_agent import CodingAgent
from .config import Config
from .coordinate_agent import CoordinateAgent
from .evaluator import Evaluator
from .evolution_agent import EvolutionAgent
from .experiment_agent import ExperimentAgent
from .flow_graph import FlowGraph
from .ideation_agent import IdeationAgent
from .research_database import ResearchDatabase
from .search import LiteratureSearch
from .utils import (Solution, LLMClient, CountingEvaluator, CountingModel,
                    Evaluation, Idea, Model, Source, Usage)


class COPScientist:
    def __init__(self, config: Config, model: Model | None = None,
                 search: LiteratureSearch | None = None,
                 checkpoint: str | None = None):
        self.config = config
        self.model = model
        self.search = search or LiteratureSearch()
        self.checkpoint = checkpoint
        self.tree = FlowGraph(config.task.objective)
        self.root_pool: list[str] = []
        self.completed_rounds = 0
        self.usage = Usage()
        self.progress: list[dict] = []
        self.coordinates: list[CoordinateAgent] = []
        self.ideation: IdeationAgent | None = None
        self.evolution: EvolutionAgent | None = None

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f">>>[COPScientist] {message}")

    def _model_for(self, role: str) -> Model:
        if self.model is not None:
            model = self.model
        else:
            settings = self.config.role_models.get(role, {})
            model = LLMClient(
                config=self.config,
                llm_provider=str(settings.get("llm_provider", self.config.llm_provider)),
                model_name=str(settings.get("model", self.config.model)),
                base_url=settings.get("base_url", self.config.base_url),
                api_key_env=settings.get("api_key_env", self.config.api_key_env),
                temperature=float(settings.get("temperature", 1.3 if role == "ideation" else 0.4)),
                max_retries=int(settings.get("max_retries", self.config.model_max_retries)),
            )
        return CountingModel(model, self.usage, role)

    def _checkpoint_path(self, name: str) -> Path:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Checkpoint must be a single directory name")
        return self.config.project_root / "checkpoints" / name

    def save(self, name: str | None = None) -> Path:
        """Snapshot config, all agent state, solution code, and SQLite."""
        if not self.root_pool:
            raise RuntimeError("No research state is available to checkpoint")
        name = name or (datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
                        + "_copscientist_" + uuid.uuid4().hex[:6])
        destination = self._checkpoint_path(name)
        if destination.exists():
            raise FileExistsError(destination)
        stage = destination.parent / f".{name}.{uuid.uuid4().hex[:8]}.tmp"
        stage.mkdir(parents=True, exist_ok=False)
        try:
            (stage / "config.json").write_text(
                json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            state = {"completed_rounds": self.completed_rounds,
                     "solutions": [solution.to_dict() for solution in self.tree.nodes.values()],
                     "root_pool": self.root_pool,
                     "usage": asdict(self.usage), "progress": self.progress,
                     "coordinates": [agent.state_dict() for agent in self.coordinates]}
            (stage / "state.json").write_text(
                json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            if self.ideation is not None:
                (stage / "ideation_agent_state.json").write_text(
                    json.dumps(self.ideation.state_dict(), indent=4), encoding="utf-8")
            source = self.config.run_path / "solutions"
            if source.is_dir():
                shutil.copytree(source, stage / "solutions")
            literature = self.config.run_path / "literature_review.json"
            if literature.is_file():
                shutil.copyfile(literature, stage / "literature_review.json")
            temporary_db = None
            if self.evolution is None:
                temporary_db = ResearchDatabase(self.config.run_path / "memory.sqlite3")
            try:
                database = self.evolution.database if self.evolution is not None else temporary_db
                database.backup(stage / "memory.sqlite3")
            finally:
                if temporary_db is not None:
                    temporary_db.close()
            stage.rename(destination)
            self._log(f"Checkpoint saved: {destination}")
        except Exception:
            checkpoint_root = destination.parent.resolve()
            if stage.exists() and stage.resolve().is_relative_to(checkpoint_root):
                shutil.rmtree(stage)
            raise
        return destination

    def _restore(self) -> dict:
        source = self._checkpoint_path(self.checkpoint)
        state = json.loads((source / "state.json").read_text(encoding="utf-8"))
        saved_config = Config.from_dict(json.loads((source / "config.json").read_text(encoding="utf-8")))
        if saved_config.task != self.config.task or saved_config.run_path != self.config.run_path:
            raise ValueError("Checkpoint task and output directory must match the active config")
        solution_root = (self.config.run_path / "solutions").resolve()
        for payload in state["solutions"]:
            code_path = Path(payload["code_path"]).resolve()
            if not code_path.is_relative_to(solution_root):
                raise ValueError(f"Checkpoint solution path is outside the run: {code_path}")
            if not (source / "solutions" / code_path.relative_to(solution_root)).is_file():
                raise FileNotFoundError(f"Checkpoint solution code missing: {code_path}")
        shutil.copyfile(source / "memory.sqlite3", self.config.run_path / "memory.sqlite3")
        if (source / "solutions").is_dir():
            shutil.copytree(source / "solutions", self.config.run_path / "solutions",
                            dirs_exist_ok=True)
        if (source / "literature_review.json").is_file():
            shutil.copyfile(source / "literature_review.json",
                            self.config.run_path / "literature_review.json")
        self.tree = FlowGraph(self.config.task.objective)
        for payload in state["solutions"]:
            payload["idea"] = Idea(**payload["idea"])
            payload["evaluation"] = Evaluation(**payload["evaluation"])
            self.tree.add(Solution(**payload))
        self.root_pool = list(state["root_pool"])
        self.completed_rounds = int(state["completed_rounds"])
        self.usage = Usage(**state["usage"])
        self.progress = list(state.get("progress", []))
        self._log(f"Checkpoint loaded: {source}; completed rounds: {self.completed_rounds}")
        return state

    def _write_progress(self) -> None:
        run_path = self.config.run_path
        (run_path / "results_detailed.json").write_text(
            json.dumps(self.progress, ensure_ascii=False, indent=2), encoding="utf-8")
        best_so_far = []
        best_score = None
        for entry in self.progress:
            score = entry["best_score"]
            if best_score is None or (score < best_score if self.config.task.objective == "min"
                                      else score > best_score):
                best_so_far.append(entry)
                best_score = score
        (run_path / "results.json").write_text(
            json.dumps(best_so_far, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self) -> Solution | None:
        config = self.config
        config.problem_path  # validate before creating outputs
        config.run_path.mkdir(parents=True, exist_ok=True)
        restored = self._restore() if self.checkpoint else None
        (config.run_path / "resolved_config.json").write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.evolution = EvolutionAgent(config.run_path / "memory.sqlite3")
        try:
            evaluator = Evaluator(config)
            experiment = ExperimentAgent(
                config, self._model_for("experiment_reflect"),
                CountingEvaluator(evaluator, self.usage, "experiment"),
                self._model_for("experiment_summarize"))
            ideation = IdeationAgent(config, self._model_for("ideation"), self.search)
            self.ideation = ideation
            coding = CodingAgent(config, self._model_for("coding"),
                                 CountingEvaluator(evaluator, self.usage, "coding"))
            if restored:
                idea_state = self._checkpoint_path(self.checkpoint) / "ideation_agent_state.json"
                if idea_state.is_file():
                    ideation.load_state(json.loads(idea_state.read_text(encoding="utf-8")))
                review_path = config.run_path / "literature_review.json"
                if review_path.is_file():
                    review = json.loads(review_path.read_text(encoding="utf-8"))
                    ideation.review = review.get("review", {})
                    ideation.sources = [Source(**item) for item in review.get("sources", [])]
            else:
                if config.search_enabled:
                    ideation.model = self._model_for("literature_review")
                    sources, errors = ideation.prepare_review()
                    ideation.model = self._model_for("ideation")
                else:
                    sources, errors = [], []
                self.evolution.record_sources(sources)
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
                baseline = Solution(
                    id=baseline_id, round=0, depth=0, parent_id=None,
                    coordinate_agent_id=0, solution_count=0,
                    idea=Idea(title="Baseline", rationale="Provided seed implementation",
                              algorithm="Existing implementation", expected_effect="Reference score"),
                    code=seed_path.read_text(encoding="utf-8"))
                experiment.evaluate_solution(baseline)
                self.tree.add(baseline)
                self.evolution.record_solution(config.task.name, baseline)
                if config.evolution_enabled:
                    self.evolution.distill(config.task.name, baseline, None, config.task.objective)
                if baseline.evaluation.status != "ok":
                    raise RuntimeError(f"Baseline failed: {baseline.evaluation.feedback}")
                self.root_pool = [baseline.id]
            self.coordinates = [CoordinateAgent(i, config, ideation, coding, experiment,
                                                self.evolution, self.tree,
                                                self._model_for("coordinate"))
                                for i in range(config.coordinate_agents)]
            if restored:
                for agent, state in zip(self.coordinates, restored.get("coordinates", [])):
                    agent.load_state(state)
            last_save_time = time.monotonic()
            for round_number in range(self.completed_rounds + 1, config.rounds + 1):
                if (self.usage.total_responses > config.max_evolutions
                        and self.usage.total_evaluations > config.max_evolutions):
                    break
                roots = [self.tree.nodes[identifier] for identifier in self.root_pool
                         if self.tree.nodes[identifier].evaluation.status == "ok"]
                roots.sort(key=lambda solution: solution.evaluation.score,
                           reverse=config.task.objective == "max")
                if not roots:
                    raise RuntimeError("No evaluated research root is available")
                self._log(f"Round {round_number} starts; best root score: {roots[0].evaluation.score}")
                new_leaves = []
                for agent in self.coordinates:
                    new_leaves.extend(agent.research_round(round_number,
                                                           roots[agent.number % len(roots)]))
                self.root_pool.extend(leaf.id for leaf in new_leaves
                                      if leaf.id not in self.root_pool)
                self.completed_rounds = round_number
                best = self.tree.best()
                self.progress.append({"iteration": round_number,
                                      "total_responses": self.usage.total_responses,
                                      "total_function_evals": self.usage.total_evaluations,
                                      "best_obj_overall": best.evaluation.score,
                                      "best_id": best.id, "best_score": best.evaluation.score})
                self._write_progress()
                self._log(f"Round {round_number} complete; best score: {best.evaluation.score}; "
                          f"model responses: {self.usage.total_responses}; "
                          f"evaluations: {self.usage.total_evaluations}")
                if (config.autosave_interval_minutes == 0 or
                        time.monotonic() - last_save_time >= config.autosave_interval_minutes * 60):
                    self.save()
                    last_save_time = time.monotonic()
            best = self.tree.best()
            validation = (experiment.evaluate(Path(best.code_path), validation=True,
                                              callbacks=best.evaluation.callbacks)
                          if best else None)
            (config.run_path / "result.json").write_text(json.dumps({
                "task": config.task.name, "best": best.to_dict() if best else None,
                "validation": asdict(validation) if validation else None,
                "solutions": [node.to_dict() for node in self.tree.nodes.values()],
                "root_pool": self.root_pool, "completed_rounds": self.completed_rounds,
                "usage": asdict(self.usage),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            return best
        finally:
            self.evolution.close()
            self.evolution = None
