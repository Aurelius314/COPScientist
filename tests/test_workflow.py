from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import closing
from pathlib import Path

from copscientist.config import Config, PROJECT_ROOT, load_config
from copscientist.coding_agent import CodingAgent
from copscientist.coordinate_agent import CoordinateAgent
from copscientist.core import COPScientist
from copscientist.experiment_agent import ExperimentAgent
from copscientist.flow_graph import FlowGraph
from copscientist.search import LiteratureSearch
from copscientist.utils import load_prompt, render_prompt, update_code
from copscientist.utils import Candidate, Evaluation, Idea, Task, experiment_allocation
from copscientist.evaluator import Evaluator, load_instance, validate


class FakeSearch:
    def review(self, literature_queries, web_queries, limit, required):
        return [], []


class FakeModel:
    def __init__(self):
        self.coding_calls = 0

    def complete(self, system, user):
        if "Ideation Agent" in system:
            return json.dumps({"ideas": [{"title": "Nearest neighbor", "rationale": "Local distance",
                                          "algorithm": "Choose the closest unvisited city",
                                          "expected_effect": "Comparable baseline", "sources": []}]})
        if "Coding Agent" in system:
            self.coding_calls += 1
            if self.coding_calls == 1:
                return "def solve(instance):\n    return []\n"
            return "def solve(instance):\n    return list(range(len(instance['coordinates'])))\n"
        if "Experiment Agent" in system:
            return json.dumps({"diagnosis": "Feasible", "evidence": "Measured gap",
                               "next_step": "Try a better city ordering"})
        raise AssertionError(system)


class RevisionModel:
    def complete(self, system, user):
        if "Ideation Agent" in system:
            return json.dumps({"ideas": [{"title": "Local route", "rationale": "Short edges",
                                          "algorithm": "Try nearest neighbor", "expected_effect": "Shorter tour",
                                          "sources": []}]})
        if "Coding Agent" in system:
            return "def solve(instance):\n    return list(range(len(instance['coordinates'])))\n"
        if "Experiment Agent" in system and "Examine the latest measured result" in system:
            return json.dumps({"action": "update_code", "reason": "Try nearest neighbor",
                               "code": "def solve(instance):\n    coords = instance['coordinates']\n"
                                       "    remaining = set(range(1, len(coords)))\n    tour = [0]\n"
                                       "    while remaining:\n        x, y = coords[tour[-1]]\n"
                                       "        nxt = min(remaining, key=lambda j: (coords[j][0]-x)**2 + (coords[j][1]-y)**2)\n"
                                       "        tour.append(nxt)\n        remaining.remove(nxt)\n    return tour\n"})
        if "Experiment Agent" in system:
            return json.dumps({"diagnosis": "Measured", "evidence": "Tour length",
                               "next_step": "Continue from best"})
        raise AssertionError(system)


class WorkflowTests(unittest.TestCase):
    def test_optional_callbacks_execute_inside_problem_evaluator(self):
        class CallbackModel:
            def complete(self, system, user):
                if "Examine the latest measured result" in system:
                    return json.dumps({"action": "update_callbacks", "reason": "Inspect one event",
                                       "callbacks": "class Callbacks:\n"
                                                    "    def on_step_end(self, step, metrics):\n"
                                                    "        print('callback fired', step, metrics['cost'])\n",
                                       "terminate_after_update": True})
                return json.dumps({"summary": "Callback observed the event."})

        name = f"callback_{uuid.uuid4().hex}"
        problem = PROJECT_ROOT / "problems" / "heuristics" / name
        output = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        problem.mkdir(parents=True)
        output.mkdir(parents=True)
        try:
            (problem / "eval.py").write_text(
                "if 'Callbacks' in globals():\n"
                "    Callbacks().on_step_end(1, {'cost': 2})\n"
                "print('Average: 1')\n", encoding="utf-8")
            (problem / "callbacks_description.txt").write_text(
                "Callbacks.on_step_end(step, metrics) is invoked once.", encoding="utf-8")
            config = Config(task=Task(name=name, kind="heuristic",
                                      problem_dir=f"problems/heuristics/{name}",
                                      objective="min", entry_function="solve",
                                      train_args=["1", ".", "train"],
                                      validation_args=["1", ".", "val"]),
                            output_dir=output, experiment_repeats=1, verbose=False)
            code = output / "candidate" / "gpt.py"
            code.parent.mkdir()
            code.write_text("def solve(instance):\n    return []\n", encoding="utf-8")
            agent = ExperimentAgent(config, CallbackModel())
            initial = agent.evaluate(code)
            self.assertEqual(initial.status, "ok")
            best, history, summary = agent.run(
                Idea("Inspect", "See event", "Callback", "Evidence"), code, initial, 2.0)
            self.assertEqual(best.score, 1.0)
            self.assertEqual(len(history), 2)
            self.assertIn("callback fired", history[-1]["evaluation"]["feedback"])
            self.assertEqual(summary, "Callback observed the event.")
        finally:
            self.assertTrue(problem.resolve().is_relative_to((PROJECT_ROOT / "problems").resolve()))
            shutil.rmtree(problem)
            shutil.rmtree(output)

    def test_dynamic_experiment_allocation_matches_oragent_rule(self):
        self.assertEqual(experiment_allocation(100, 90, "min", 3, 1, 12, .05, 2, False), 9)
        self.assertEqual(experiment_allocation(100, 110, "min", 3, 1, 12, .05, 2, False), 1)
        self.assertEqual(experiment_allocation(100, 100, "min", 3, 1, 12, .05, 2, True), 2)

    def test_root_retries_invalid_code_before_experiment(self):
        class Ideas:
            model = None

            def propose(self, parent, memory, count, tree, reflection):
                return [Idea("Route", "Test", "Order", "Shorter")]

        class Coding:
            calls = 0

            def run(self, idea, parent, path, reflection):
                self.calls += 1
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("def solve(instance):\n    return []\n", encoding="utf-8")
                return Evaluation(status="failed" if self.calls == 1 else "ok",
                                  score=None if self.calls == 1 else 9.0)

        class Experiment:
            calls = 0

            def run(self, idea, path, evaluation, parent_score, reflection, max_repeats=None):
                self.calls += 1
                return evaluation, [], "Measured result"

        class Memory:
            def retrieve(self, task):
                return []

            def record_candidate(self, task, candidate):
                pass

            def distill(self, task, candidate, parent, objective):
                pass

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.max_depth = 1
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            seed = temp / "seed.py"
            seed.write_text("def solve(instance):\n    return []\n", encoding="utf-8")
            root = Candidate("root", 0, 0, None, Idea("Seed", "Seed", "Seed", "Seed"),
                             str(seed), Evaluation("ok", 10.0))
            tree = FlowGraph("min")
            tree.add(root)
            coding, experiment = Coding(), Experiment()
            leaves = CoordinateAgent(0, config, Ideas(), coding, experiment,
                                     Memory(), tree).research_round(1, root)
            self.assertEqual(coding.calls, 2)
            self.assertEqual(experiment.calls, 1)
            self.assertEqual(len(leaves), 1)
            self.assertEqual(len(tree.nodes), 2)
        finally:
            shutil.rmtree(temp)

    def test_checkpoint_restores_round_state_and_usage(self):
        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.task.instances = ["tsplib_euc_2d/eil51.tsp.gz"]
        config.task.validation_instances = ["tsplib_euc_2d/eil76.tsp.gz"]
        config.rounds = 1
        config.max_depth = 1
        config.ideas_per_round = 1
        config.experiment_repeats = 1
        config.dynamic_experiment_allocation = False
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        checkpoint = None
        try:
            config.output_dir = temp
            scientist = COPScientist(config, FakeModel(), FakeSearch())
            scientist.run()
            checkpoint = scientist.save(f"test_{uuid.uuid4().hex}")
            first = json.loads((temp / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(first["completed_rounds"], 1)
            self.assertEqual(len(first["root_pool"]), 2)
            self.assertTrue((checkpoint / "memory.sqlite3").is_file())
            baseline_code = Path(first["candidates"][0]["code_path"])
            baseline_code.unlink()
            self.assertFalse(baseline_code.exists())
            resumed_config = Config.from_dict(json.loads((checkpoint / "config.json").read_text(encoding="utf-8")))
            resumed_config.rounds = 2
            COPScientist(resumed_config, FakeModel(), FakeSearch(), checkpoint.name).run()
            second = json.loads((temp / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(second["completed_rounds"], 2)
            self.assertTrue(baseline_code.is_file())
            self.assertEqual(len(second["candidates"]), 3)
            self.assertGreater(sum(second["usage"]["responses"].values()),
                               sum(first["usage"]["responses"].values()))
            self.assertEqual(len(json.loads((temp / "results_detailed.json").read_text(encoding="utf-8"))), 2)
        finally:
            shutil.rmtree(temp)
            if checkpoint is not None:
                self.assertEqual(checkpoint.parent, PROJECT_ROOT / "checkpoints")
                shutil.rmtree(checkpoint)

    def test_evolution_budget_stops_after_completed_round(self):
        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.task.instances = ["tsplib_euc_2d/eil51.tsp.gz"]
        config.task.validation_instances = ["tsplib_euc_2d/eil76.tsp.gz"]
        config.rounds = 2
        config.max_depth = 1
        config.ideas_per_round = 1
        config.experiment_repeats = 1
        config.max_evolutions = 0
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            COPScientist(config, FakeModel(), FakeSearch()).run()
            result = json.loads((temp / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["completed_rounds"], 1)
            self.assertGreater(sum(result["usage"]["responses"].values()), 0)
            self.assertGreater(sum(result["usage"]["evaluations"].values()), 0)
        finally:
            shutil.rmtree(temp)

    def test_coding_retries_syntax_error_with_feedback(self):
        class RetryModel:
            def __init__(self):
                self.users = []

            def complete(self, system, user):
                self.users.append(user)
                if len(self.users) == 1:
                    return "def solve(instance):\n    return (\n"
                return "def solve(instance):\n    return []\n"

        class AcceptEvaluator:
            def evaluate(self, path):
                return Evaluation(status="ok", score=1.0)

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            model = RetryModel()
            result = CodingAgent(config, model, AcceptEvaluator()).run(
                Idea("Route", "Test", "Order", "Shorter"), None, temp / "candidate.py")
            self.assertEqual(result.status, "ok")
            self.assertEqual(len(model.users), 2)
            self.assertIn("'(' was never closed", model.users[1])
        finally:
            shutil.rmtree(temp)

    def test_agent_prompts_are_external_and_renderable(self):
        for stage in ("literature_review", "idea_generation", "code_generation",
                      "code_debugging", "experiment_reflection", "experiment_summary",
                      "long_term_reflection"):
            self.assertTrue(load_prompt(f"system_{stage}_copscientist.txt"))
            self.assertIn("example", render_prompt(f"user_{stage}_copscientist.txt", "example"))
        with self.assertRaises(ValueError):
            load_prompt("../config.yaml")

    def test_search_replace_code_revision(self):
        source = "def solve(instance):\n    return []\n"
        diff = "<<<<<<< SEARCH\n    return []\n=======\n    return [0]\n>>>>>>> REPLACE"
        self.assertIn("return [0]", update_code(source, diff))

    def test_experiment_restores_best_version_after_failed_revision(self):
        class BadRevisionModel:
            def complete(self, system, user):
                if "Examine the latest measured result" in system:
                    return json.dumps({"action": "update_code", "code": "def solve(instance):\n    return []\n"})
                return json.dumps({"diagnosis": "Revision failed", "evidence": "Infeasible route",
                                   "next_step": "Use previous version"})

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.experiment_repeats = 2
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            path = temp / "candidate.py"
            original = "def solve(instance):\n    return list(range(len(instance['coordinates'])))\n"
            path.write_text(original, encoding="utf-8")
            agent = ExperimentAgent(config, BadRevisionModel())
            initial = Evaluation(status="ok", score=10.0)
            result, history, _ = agent.run(
                Idea("Route", "Test", "Order", "Shorter"), path, initial, 12.0)
            self.assertEqual(result.score, 10.0)
            self.assertEqual(len(history), 3)
            self.assertEqual(history[1]["evaluation"]["status"], "failed")
            self.assertEqual(path.read_text(encoding="utf-8"), original)
        finally:
            shutil.rmtree(temp)

    def test_solver_rejects_invalid_tour_and_packing(self):
        tsp = load_instance("tsp", PROJECT_ROOT / "problems/solvers/tsp/tsplib_euc_2d/berlin52.tsp.gz")
        with self.assertRaises(ValueError):
            validate("tsp", tsp, [0] * len(tsp["coordinates"]))
        packing = {"bin_size": [10, 10, 10], "items": [[5, 5, 5], [5, 5, 5]]}
        with self.assertRaises(ValueError):
            validate("3d-bpp", packing, [[0, 0, 0, 0], [0, 0, 0, 0]])

    def test_literature_search_preserves_provenance(self):
        def fetch(url, headers=None):
            if "crossref" in url:
                return {"message": {"items": [{"title": ["A scheduling paper"], "DOI": "10.1/a",
                                                "published": {"date-parts": [[2020]]}}]}}
            return {"web": {"results": [{"title": "Routing guide", "url": "https://example.org/route"}]}}
        search = LiteratureSearch(fetch=fetch)
        import os
        previous = os.environ.get("BRAVE_SEARCH_API_KEY")
        os.environ["BRAVE_SEARCH_API_KEY"] = "test"
        try:
            sources, errors = search.review(["scheduling"], ["routing"], 2, True)
        finally:
            if previous is None:
                del os.environ["BRAVE_SEARCH_API_KEY"]
            else:
                os.environ["BRAVE_SEARCH_API_KEY"] = previous
        self.assertEqual(errors, [])
        self.assertEqual({s.kind for s in sources}, {"paper", "web"})
        self.assertEqual(sources[0].url, "https://doi.org/10.1/a")

    def test_full_research_round_persists_tree_and_memory(self):
        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.task.instances = ["tsplib_euc_2d/eil51.tsp.gz"]
        config.task.validation_instances = ["tsplib_euc_2d/eil76.tsp.gz"]
        config.rounds = 1
        config.ideas_per_round = 1
        config.max_depth = 1
        config.debug_attempts = 1
        config.experiment_repeats = 1
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            model = FakeModel()
            best = COPScientist(config, model, FakeSearch()).run()
            result = json.loads((temp / "result.json").read_text(encoding="utf-8"))
            resolved = json.loads((temp / "resolved_config.json").read_text(encoding="utf-8"))
            self.assertEqual(resolved["task"]["name"], "tsp")
            self.assertEqual(len(result["candidates"]), 2)
            self.assertEqual(result["candidates"][1]["parent_id"], result["candidates"][0]["id"])
            self.assertEqual(result["candidates"][1]["evaluation"]["status"], "ok")
            self.assertEqual(model.coding_calls, 2)
            self.assertEqual(result["validation"]["status"], "ok")
            self.assertIsNotNone(best.evaluation.score)
            import sqlite3
            with closing(sqlite3.connect(temp / "memory.sqlite3")) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 0)
        finally:
            shutil.rmtree(temp)

    def test_heuristic_candidate_is_isolated_from_seed(self):
        config = load_config(PROJECT_ROOT / "config.yaml", "tsp_constructive")
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            candidate = temp / "gpt.py"
            candidate.write_text("def select_next_node(*args, **kwargs):\n    raise RuntimeError('candidate marker')\n", encoding="utf-8")
            result = ExperimentAgent(config).evaluate(candidate)
            self.assertEqual(result.status, "failed")
            self.assertIn("candidate marker", result.feedback)
            self.assertNotIn("candidate marker", (config.problem_path / config.task.seed_file).read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(temp)

    def test_heuristic_baseline_evaluates_and_keeps_raw_output(self):
        config = load_config(PROJECT_ROOT / "config.yaml", "tsp_constructive")
        config.rounds = 0
        config.verbose = False
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            baseline = COPScientist(config, FakeModel(), FakeSearch()).run()
            result = json.loads((temp / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(baseline.evaluation.status, "ok")
            self.assertEqual(result["validation"]["status"], "ok")
            self.assertTrue(Path(baseline.evaluation.output_path).is_file())
            self.assertEqual(result["root_pool"], [baseline.id])
            self.assertEqual(result["usage"]["responses"], {})
        finally:
            shutil.rmtree(temp)

    def test_experiment_revision_and_second_tree_depth(self):
        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.task.instances = ["tsplib_euc_2d/eil51.tsp.gz"]
        config.task.validation_instances = ["tsplib_euc_2d/eil76.tsp.gz"]
        config.rounds = 1
        config.ideas_per_round = 1
        config.max_depth = 2
        config.debug_attempts = 0
        config.experiment_repeats = 2
        config.dynamic_experiment_allocation = False
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            COPScientist(config, RevisionModel(), FakeSearch()).run()
            result = json.loads((temp / "result.json").read_text(encoding="utf-8"))
            candidates = result["candidates"]
            self.assertEqual(len(candidates), 3)
            self.assertEqual(candidates[2]["parent_id"], candidates[1]["id"])
            self.assertEqual(len(candidates[1]["intermediate"]), 3)
            round_tree = json.loads((temp / "details" / "coordinate_0_round_1_tree.json").read_text(encoding="utf-8"))
            self.assertEqual(len(round_tree["nodes"]), 2)
            self.assertIn(candidates[1]["id"], result["root_pool"])
            self.assertNotIn(candidates[2]["id"], result["root_pool"])
            initial = candidates[1]["intermediate"][0]["evaluation"]["score"]
            self.assertLess(candidates[1]["evaluation"]["score"], initial)
            self.assertIn("remaining =", Path(candidates[1]["code_path"]).read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(temp)

    def test_configuration_precedence(self):
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            global_path = temp / "config.yaml"
            global_path.write_text('problem: tsp_constructive\nrun:\n  rounds: 5\n  max_depth: null\n', encoding="utf-8")
            config = load_config(global_path, "tsp", {"rounds": 7, "output_dir": Path("runs/custom")})
            self.assertEqual(config.task.name, "tsp")
            self.assertEqual(config.task.objective, "min")
            self.assertEqual(config.rounds, 7)
            self.assertIsNone(config.max_depth)
            self.assertEqual(config.run_path, PROJECT_ROOT / "runs/custom")
        finally:
            shutil.rmtree(temp)


if __name__ == "__main__":
    unittest.main()
