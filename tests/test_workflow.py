from __future__ import annotations

import json
import shutil
import subprocess
import unittest
import uuid
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from copscientist.config import Config, PROJECT_ROOT, load_config
from copscientist.coding_agent import CodingAgent
from copscientist.coordinate_agent import CoordinateAgent
from copscientist.core import COPScientist
from copscientist.experiment_agent import ExperimentAgent
from copscientist.flow_graph import FlowGraph
from copscientist.ideation_agent import IdeationAgent
from copscientist.search import LiteratureSearch
from copscientist.utils import load_prompt, parse_code, render_prompt, update_code
from copscientist.utils import Evaluation, Idea, LLMClient, Solution, Source, Task, experiment_allocation
from copscientist.evaluator import (Evaluator, _average_metrics, _parse_eval_result,
                                    load_instance, validate)


class FakeSearch:
    def review(self, literature_queries, web_queries, limit, required):
        return [], []


def experiment_reply(termination="yes", thinking="Measured result", diff=None, callbacks=None):
    parts = [f"<thinking>{thinking}</thinking>"]
    if diff is not None:
        parts.append(f"```diff\n{diff}\n```")
    if callbacks is not None:
        parts.append(f"```python\n{callbacks}\n```")
    parts.append(f'```json\n{{"termination": "{termination}"}}\n```')
    return "\n".join(parts)


class FakeModel:
    def __init__(self):
        self.coding_calls = 0

    def complete(self, system, user):
        if "research survey assistant" in system:
            return json.dumps({"claims": [{"claim": "Use measured local costs",
                                            "source_urls": ["https://example.org/paper"]}],
                               "transferable_methods": [], "limitations": [],
                               "sufficient": True, "follow_up_queries": []})
        if "Ideation Agent" in system:
            return json.dumps([{"title": "Nearest neighbor", "rationale": "Local distance",
                                "algorithm": "Choose the closest unvisited city",
                                "expected_effect": "Comparable baseline", "sources": []}])
        if "Coding Agent" in system:
            self.coding_calls += 1
            if self.coding_calls == 1:
                return "def solve(instance):\n    return []\n"
            return "def solve(instance):\n    return list(range(len(instance['coordinates'])))\n"
        if "Experiment Agent" in system:
            if "final report" not in system.lower():
                return experiment_reply(thinking="No justified local revision")
            return json.dumps({"diagnosis": "Feasible", "evidence": "Measured gap",
                               "next_step": "Try a better city ordering"})
        raise AssertionError(system)


class RevisionModel:
    def complete(self, system, user):
        if "Ideation Agent" in system:
            return json.dumps([{"title": "Local route", "rationale": "Short edges",
                                "algorithm": "Try nearest neighbor", "expected_effect": "Shorter tour",
                                "sources": []}])
        if "Coding Agent" in system:
            return "def solve(instance):\n    return list(range(len(instance['coordinates'])))\n"
        if "Experiment Agent" in system and "Examine the latest measured result" in system:
            if "remaining = set" in user:
                return experiment_reply(thinking="The local change has been measured")
            old = "def solve(instance):\n    return list(range(len(instance['coordinates'])))"
            new = ("def solve(instance):\n    coords = instance['coordinates']\n"
                   "    remaining = set(range(1, len(coords)))\n    tour = [0]\n"
                   "    while remaining:\n        x, y = coords[tour[-1]]\n"
                   "        nxt = min(remaining, key=lambda j: (coords[j][0]-x)**2 + (coords[j][1]-y)**2)\n"
                   "        tour.append(nxt)\n        remaining.remove(nxt)\n    return tour")
            return experiment_reply("no", "Try nearest neighbor",
                                    f"<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE")
        if "Experiment Agent" in system:
            return json.dumps({"diagnosis": "Measured", "evidence": "Tour length",
                               "next_step": "Continue from best"})
        raise AssertionError(system)


class WorkflowTests(unittest.TestCase):
    def test_ideation_coordinated_generation_retries_and_counts_responses(self):
        class Model:
            def __init__(self):
                self.calls = []

            def complete(self, system, user):
                self.calls.append((system, user))
                if len(self.calls) == 1:
                    return "invalid JSON"
                return json.dumps([{"title": "Change insertion", "rationale": "Measured gap",
                                    "algorithm": "Insert the cheapest next node",
                                    "expected_effect": "Lower cost", "sources": ["https://known", "https://unknown"]},
                                   {"title": "Local search", "rationale": "Long edges",
                                    "algorithm": "Apply 2-opt", "expected_effect": "Lower cost",
                                    "sources": []}])

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.verbose = False
        model = Model()
        agent = IdeationAgent(config, model, FakeSearch())
        agent.sources = [Source(url="https://known", title="Paper", kind="literature",
                                query="search", abstract="Measured result",
                                content_level="abstract")]
        parent = Solution(idea=Idea("Seed", "Baseline", "Original", "Reference"),
                          code="def solve(instance): return []\n",
                          evaluation=Evaluation(status="ok", score=10.0))
        ideas = agent.run(parent, [{"insight": "try cheaper edges"}], 2,
                          FlowGraph("min", root=parent), "Keep feasible tours")
        self.assertEqual(len(ideas), 2)
        self.assertEqual(ideas[0].sources, ["https://known"])
        self.assertEqual(agent.state_dict(), {"total_responses": 2, "function_evals": 0,
                                               "valid_responses": 1})
        self.assertIn("Keep feasible tours", model.calls[0][1])
        self.assertIn("try cheaper edges", model.calls[0][1])
        self.assertIn("Current Research Flow Graph", model.calls[0][1])
        self.assertIn("coordinated", (model.calls[0][0] + model.calls[0][1]).lower())

    def test_ideation_independent_generation_retries_each_idea(self):
        class Model:
            def __init__(self):
                self.calls = []

            def complete(self, system, user):
                self.calls.append((system, user))
                if len(self.calls) == 1:
                    return "not JSON"
                item = {"title": f"Idea {len(self.calls)}", "rationale": "Observed issue",
                        "algorithm": "Try one local change", "expected_effect": "Better score",
                        "sources": []}
                return json.dumps({"idea": item})

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.verbose = False
        config.ideas_coordinated_generation_disabled = True
        model = Model()
        agent = IdeationAgent(config, model, FakeSearch())
        ideas = agent.run(None, [], 2, FlowGraph("min"))
        self.assertEqual([idea.title for idea in ideas], ["Idea 2", "Idea 3"])
        self.assertEqual(agent.total_responses, 3)
        self.assertEqual(agent.valid_responses, 2)
        self.assertEqual(model.calls[0][1], model.calls[1][1])
        self.assertIn("independent", model.calls[0][0].lower())
        agent.reset()
        self.assertEqual(agent.total_responses, 0)
        self.assertEqual(agent.valid_responses, 0)

    def test_ideation_searches_current_failure_mode_before_generating_ideas(self):
        class DecisionModel:
            def __init__(self):
                self.users = []

            def complete(self, system, user):
                self.users.append(user)
                if "NEW external evidence" in system:
                    return json.dumps({"search": True, "failure_mode": "Repeated long closing edge",
                                       "queries": [{"kind": "web", "query": "TSP closing edge repair"},
                                                   {"kind": "web", "query": "TSP closing edge repair"},
                                                   {"kind": "web", "query": "discarded extra query"}]})
                return json.dumps({"claims": [{"claim": "Repair the closing edge",
                                                "source_urls": ["https://example.org/repair"]}],
                                   "transferable_methods": [], "limitations": [],
                                   "sufficient": True, "follow_up_queries": []})

        class IdeaModel:
            user = ""

            def complete(self, system, user):
                self.user = user
                return json.dumps([{"title": "Closing-edge repair",
                                    "rationale": "Parent has a long closing edge",
                                    "algorithm": "Repair the tour after construction",
                                    "expected_effect": "Reduce tour length",
                                    "sources": ["https://example.org/repair",
                                                "https://invented.example/claim"]}])

        class Search:
            queries = []

            def review(self, paper_queries, web_queries, limit, required):
                self.queries.append((paper_queries, web_queries))
                return [Source(title="Repair note", url="https://example.org/repair",
                               kind="web", query=web_queries[0], abstract="Search snippet")], []

            def read_sources(self, sources, max_pages=2):
                sources[0].content = "The page describes a closing-edge repair."
                sources[0].content_level = "page"
                return []

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.search_enabled = True
        config.search_idea_max_queries = 1
        config.verbose = False
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            parent = Solution(idea=Idea("Seed", "Baseline", "Tour", "Reference"),
                              code="def solve(instance): return []\n",
                              evaluation=Evaluation(status="ok", score=120.0,
                                                    feedback="Long closing edge"),
                              summary="Repeated long closing edge after construction")
            search, decision, idea_model = Search(), DecisionModel(), IdeaModel()
            agent = IdeationAgent(config, idea_model, search, decision)
            ideas = agent.run(parent, [], 1, FlowGraph("min", root=parent))
            self.assertEqual(search.queries, [([], ["TSP closing edge repair"])])
            self.assertIn("Repeated long closing edge", decision.users[0])
            self.assertIn("current_parent_review", idea_model.user)
            self.assertIn("Repair the closing edge", idea_model.user)
            self.assertEqual(ideas[0].sources, ["https://example.org/repair"])
            files = list((temp / "ideation_surveys").glob("*.json"))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text(encoding="utf-8"))["parent_id"],
                             parent.id)
        finally:
            shutil.rmtree(temp)

    def test_ideation_skips_local_search_when_parent_has_enough_evidence(self):
        class DecisionModel:
            def complete(self, system, user):
                return json.dumps({"search": False, "failure_mode": "Already understood",
                                   "queries": []})

        class IdeaModel:
            def complete(self, system, user):
                return json.dumps([{"title": "Improve", "rationale": "Measured gap",
                                    "algorithm": "Adjust the construction rule",
                                    "expected_effect": "Lower score", "sources": []}])

        class Search:
            def review(self, *args):
                raise AssertionError("Search should not run")

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp_constructive")
        config.search_enabled = True
        config.verbose = False
        agent = IdeationAgent(config, IdeaModel(), Search(), DecisionModel())
        parent = Solution(idea=Idea("Seed", "Baseline", "Rule", "Reference"),
                          evaluation=Evaluation(status="ok", score=1.0))
        ideas = agent.run(parent, [], 1, FlowGraph("min", root=parent))
        self.assertEqual(len(ideas), 1)

    def test_solver_ideation_prompt_preserves_feasibility_and_runtime_requirements(self):
        class Model:
            user = ""

            def complete(self, system, user):
                self.user = user
                return json.dumps([{"title": "Feasible insertion", "rationale": "Tour gap",
                                    "algorithm": "Construct then repair a valid tour",
                                    "expected_effect": "Lower cost", "sources": []}])

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.verbose = False
        model = Model()
        agent = IdeationAgent(config, model, FakeSearch())
        agent.run(None, [], 1, FlowGraph("min"))
        self.assertIn("valid solution", model.user)
        self.assertIn("runtime cost", model.user)
        self.assertIn("Implement `solve`", model.user)

    def test_llm_client_provider_chat_retry_and_multi_chat(self):
        def response(content):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=content))])

        with patch("openai.OpenAI") as factory, patch("copscientist.utils.time.sleep"):
            create = factory.return_value.chat.completions.create
            create.side_effect = [RuntimeError("temporary"), response("ready")]
            client = LLMClient(config={"llm": {"llm_provider": "qwen", "model": "qwen-test",
                                               "max_retries": 2}}, api_key="test-key")
            self.assertEqual(client.complete("system", "user"), "ready")
            self.assertEqual(factory.call_args.kwargs["base_url"],
                             "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.assertEqual(create.call_args.kwargs["messages"][0]["role"], "system")
            create.side_effect = lambda **kwargs: response(kwargs["messages"][-1]["content"])
            self.assertEqual(client.multi_chat([
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}]]), ["first", "second"])
            self.assertEqual(client.multi_chat([{"role": "user", "content": "again"}], n=2),
                             ["again", "again"])
            with self.assertRaises(ValueError):
                client.multi_chat([], n=1)

    def test_llm_client_role_provider_override(self):
        config = load_config(PROJECT_ROOT / "config.yaml", "tsp",
                             {"llm_provider": "qwen", "model": "qwen-test"})
        config.role_models = {"coding": {"llm_provider": "deepseek",
                                         "model": "deepseek-test"}}
        with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "qwen-key",
                                    "DEEPSEEK_API_KEY": "deepseek-key"}), patch("openai.OpenAI") as factory:
            scientist = COPScientist(config, search=FakeSearch())
            ideation = scientist._model_for("ideation").model
            self.assertEqual(ideation.llm_provider, "qwen")
            self.assertEqual(ideation.base_url,
                             "https://dashscope.aliyuncs.com/compatible-mode/v1")
            coding = scientist._model_for("coding").model
            self.assertEqual(coding.model_name, "deepseek-test")
            self.assertEqual(coding.base_url, "https://api.deepseek.com")
            self.assertEqual(factory.call_args.kwargs["api_key"], "deepseek-key")

    def test_solution_matches_oragent_record_interface(self):
        idea = Idea("Route", "Shorter tour", "Nearest node", "Lower cost")
        result = Evaluation(status="ok", score=3.5, metrics={20: 3.5}, features=(3,),
                            feedback="evaluated")
        solution = Solution(id="example", round=2, coordinate_agent_id=1,
                            solution_count=4, idea=idea, code="def solve(): pass\n",
                            evaluation=result, reflection="Keep this approach",
                            intermediate=[{"evaluation": {"score": 3.5, "metrics": {20: 3.5},
                                                           "features": [3], "feedback": "evaluated"},
                                           "action": {"action": "terminate"}}])
        self.assertEqual(solution.id_str(), "coordinate1_round2_count4_idexample")
        self.assertEqual(solution.score, result.score)
        self.assertEqual(solution.metrics, result.metrics)
        self.assertEqual(solution.features, result.features)
        self.assertIn("Experiment #1", solution.experiment_history())
        self.assertIn("Keep this approach", solution.idea_performance_summary_str())
        solution.score = 2.0
        self.assertEqual(solution.evaluation.score, 2.0)
        solution.evaluation = Evaluation(status="ok", score=1.0, metrics={20: 1.0})
        self.assertEqual(solution.score, 1.0)
        self.assertEqual(solution.to_dict()["code"], "def solve(): pass\n")

    def test_heuristic_result_protocol(self):
        output = ("[*] Average: 3.5\n__SANDBOX_RESULT__\n"
                  "__METRICS_START__\n{20: 3.0, 50: 4.0}\n__METRICS_END__\n"
                  "__FEATURES_START__\n(3, 4)\n__FEATURES_END__\n"
                  "__SCORE_START__\n3.5\n__SCORE_END__\n__SANDBOX_SUCCESS__\n")
        self.assertEqual(_parse_eval_result(output), ({20: 3.0, 50: 4.0}, (3, 4), 3.5))
        with self.assertRaises(ValueError):
            _parse_eval_result("[*] Average: 3.5\n")

    def test_evaluator_accepts_oragent_metrics_and_independent_features(self):
        output = ("__SANDBOX_RESULT__\n"
                  "__METRICS_START__\n{'small': {'cost': np.float64(2.5)}}\n__METRICS_END__\n"
                  "__FEATURES_START__\nNone\n__FEATURES_END__\n"
                  "__SCORE_START__\nnp.float64(1.25)\n__SCORE_END__\n"
                  "__SANDBOX_SUCCESS__\n")
        self.assertEqual(_parse_eval_result(output),
                         ({"small": {"cost": 2.5}}, None, 1.25))
        self.assertEqual(_average_metrics([{"small": {"cost": 2.0}},
                                           {"small": {"cost": 4.0}}]),
                         {"small": {"cost": 3.0}})
        with self.assertRaises(ValueError):
            _parse_eval_result(output.replace("np.float64(1.25)", "__import__('os')"))

    def test_evaluator_run_updates_solution_and_preserves_code_versions(self):
        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.task.instances = ["tsplib_euc_2d/eil51.tsp.gz"]
        directory = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        try:
            config.output_dir = directory
            code = "def solve(instance):\n    return list(range(len(instance['coordinates'])))\n"
            solution = Solution(id="solver", code=code)
            evaluator = Evaluator(config)
            raw, metrics, features, score = evaluator.run(solution)
            original_path = Path(solution.code_path)
            self.assertEqual(original_path.parent.parent.name, "solver")
            self.assertEqual(original_path.name, f"{original_path.parent.name}_solution.py")
            self.assertEqual(solution.evaluation.status, "ok")
            self.assertEqual(solution.score, score)
            self.assertEqual(solution.metrics, metrics)
            self.assertEqual(len(features), 1)
            self.assertTrue(Path(solution.output_filepath).is_file())
            self.assertIn("eil51", raw)
            solution.code = code + "\n"
            evaluator.evaluate_individual(solution)
            self.assertNotEqual(Path(solution.code_path), original_path)
            self.assertEqual(original_path.read_text(encoding="utf-8"), code)
            wrong_kind = directory / "solutions" / "heuristic" / "wrong" / "wrong_solution.py"
            wrong_kind.parent.mkdir(parents=True)
            wrong_kind.write_text(code, encoding="utf-8")
            moved = Solution(id="wrong", code_path=str(wrong_kind), code=code)
            evaluator.run(moved)
            self.assertEqual(Path(moved.code_path).parent.parent.name, "solver")
            self.assertEqual(wrong_kind.read_text(encoding="utf-8"), code)
        finally:
            shutil.rmtree(directory)

    def test_evaluator_preserves_timeout_output_and_handles_missing_code(self):
        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.task.instances = ["tsplib_euc_2d/eil51.tsp.gz"]
        directory = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        try:
            config.output_dir = directory
            evaluator = Evaluator(config)
            missing = evaluator.evaluate_individual(Solution(id="missing"))
            self.assertEqual(missing.evaluation.status, "failed")
            code_path = directory / "solution.py"
            code_path.write_text("def solve(instance): return []\n", encoding="utf-8")
            with patch.object(evaluator, "_run", side_effect=subprocess.TimeoutExpired(
                    "python", 2, output=b"partial stdout")):
                result = evaluator.evaluate(code_path)
            self.assertEqual(result.status, "failed")
            self.assertIn("partial stdout", Path(result.output_path).read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(directory)

    def test_evaluator_executes_saved_eval_snapshot(self):
        name = f"snapshot_{uuid.uuid4().hex}"
        problem = PROJECT_ROOT / "problems" / "heuristics" / name
        directory = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        problem.mkdir(parents=True)
        directory.mkdir(parents=True)
        try:
            original = ("from seed_solution import solve\n"
                        "print('__SANDBOX_RESULT__')\n"
                        "print('__METRICS_START__\\n{1: 1.0}\\n__METRICS_END__')\n"
                        "print('__FEATURES_START__\\n(1,)\\n__FEATURES_END__')\n"
                        "print('__SCORE_START__\\n1.0\\n__SCORE_END__')\n"
                        "print('__SANDBOX_SUCCESS__')\n")
            (problem / "eval.py").write_text(original, encoding="utf-8")
            config = Config(task=Task(name=name, kind="heuristic",
                                      problem_dir=f"problems/heuristics/{name}",
                                      objective="min", entry_function="solve",
                                      train_args=["1", ".", "train"]), output_dir=directory)
            evaluator = Evaluator(config)
            (problem / "eval.py").write_text("raise RuntimeError('changed')\n", encoding="utf-8")
            solution = Solution(id="snapshot", code="def solve(): return 1\n")
            raw, metrics, features, score = evaluator.run(solution)
            self.assertEqual((metrics, features, score), ({1: 1.0}, (1,), 1.0))
            self.assertIn("__SANDBOX_SUCCESS__", raw)
            copied = Path(solution.code_path).parent / f"{solution.id_str()}_eval.py"
            self.assertEqual(copied.read_text(encoding="utf-8"), original)
        finally:
            shutil.rmtree(problem)
            shutil.rmtree(directory)

    def test_named_heuristic_solution_loads_seed_and_shared_helpers(self):
        config = load_config(PROJECT_ROOT / "config.yaml", "bpp_online")
        directory = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        try:
            config.output_dir = directory
            seed = (config.problem_path / config.task.seed_file).read_text(encoding="utf-8")
            solution = Solution(id="binpacking", code=seed)
            raw, metrics, features, score = Evaluator(config).run(solution)
            self.assertEqual(solution.evaluation.status, "ok")
            self.assertEqual(list(metrics), ["train_0", "train_1"])
            self.assertEqual(score, solution.score)
            self.assertEqual(Path(solution.code_path).parent.parent.name, "heuristic")
            self.assertIn("__SANDBOX_SUCCESS__", raw)
        finally:
            shutil.rmtree(directory)

    def test_optional_callbacks_execute_inside_problem_evaluator(self):
        class CallbackModel:
            def complete(self, system, user):
                if "Examine the latest measured result" in system:
                    return experiment_reply(
                        "yes", "Inspect one event",
                        diff="<<<<<<< SEARCH\n    return []\n=======\n"
                             "    return [0]\n>>>>>>> REPLACE",
                        callbacks="class Callbacks:\n"
                                  "    def on_step_end(self, step, metrics):\n"
                                  "        print('callback fired', step, metrics['cost'])\n")
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
                "print('__SANDBOX_RESULT__')\n"
                "print('__METRICS_START__\\n{1: 1.0}\\n__METRICS_END__')\n"
                "print('__FEATURES_START__\\n(1,)\\n__FEATURES_END__')\n"
                "print('__SCORE_START__\\n1.0\\n__SCORE_END__')\n"
                "print('__SANDBOX_SUCCESS__')\n", encoding="utf-8")
            (problem / "callbacks_description.txt").write_text(
                "Callbacks.on_step_end(step, metrics) is invoked once.", encoding="utf-8")
            config = Config(task=Task(name=name, kind="heuristic",
                                      problem_dir=f"problems/heuristics/{name}",
                                      objective="min", entry_function="solve",
                                      train_args=["1", ".", "train"],
                                      validation_args=["1", ".", "val"]),
                            output_dir=output, experiment_repeats=1, verbose=False)
            code = output / "solution" / "test_solution.py"
            code.parent.mkdir()
            code.write_text("def solve(instance):\n    return []\n", encoding="utf-8")
            agent = ExperimentAgent(config, CallbackModel())
            initial = agent.evaluate(code)
            self.assertEqual(initial.status, "ok")
            best, history, summary, compact = agent.run(
                Idea("Inspect", "See event", "Callback", "Evidence"), code, initial, 2.0)
            self.assertEqual(best.score, 1.0)
            self.assertEqual(len(history), 2)
            self.assertIn("callback fired", history[-1]["evaluation"]["feedback"])
            self.assertIn("return [0]", Path(history[-1]["code_path"]).read_text(encoding="utf-8"))
            self.assertEqual(summary, "Callback observed the event.")
            self.assertEqual(compact, [])
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

            def run(self, parent, memory, count, tree, reflection):
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

            def run(self, idea, path, evaluation, parent_score, reflection,
                    max_repeats=None, parent=None):
                self.calls += 1
                return evaluation, [], "Measured result", []

        class Memory:
            def retrieve(self, task):
                return []

            def record_solution(self, task, solution):
                pass

            def distill(self, task, solution, parent, objective):
                pass

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.max_depth = 1
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            seed = temp / "seed.py"
            seed.write_text("def solve(instance):\n    return []\n", encoding="utf-8")
            root = Solution("root", 0, 0, None, Idea("Seed", "Seed", "Seed", "Seed"),
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

    def test_coordinate_debugs_invalid_solution_after_experiment(self):
        class Ideas:
            model = None

            def run(self, parent, memory, count, tree, reflection):
                return [Idea("Route", "Test", "Order", "Shorter")]

        class Coding:
            calls = 0
            debug_calls = 0

            def run(self, idea, parent, path, reflection):
                self.calls += 1
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("def solve(instance):\n    return []\n", encoding="utf-8")
                return (Evaluation(status="ok", score=9.0) if self.calls == 1
                        else Evaluation(status="failed"))

            def debug(self, idea, parent, path, evaluation, reflection):
                self.debug_calls += 1
                path.write_text("def solve(instance):\n    return [0]\n", encoding="utf-8")
                return Evaluation(status="ok", score=8.0)

        class Experiment:
            def run(self, idea, path, evaluation, parent_score, reflection,
                    max_repeats=None, parent=None):
                return evaluation, [], "", []

        class Memory:
            def retrieve(self, task):
                return []

            def record_solution(self, task, solution):
                pass

            def distill(self, task, solution, parent, objective):
                pass

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.max_depth = 2
        config.dynamic_experiment_allocation = False
        config.verbose = False
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            root = Solution(id="root", idea=Idea("Seed", "Seed", "Seed", "Seed"),
                            code="def solve(instance):\n    return []\n",
                            evaluation=Evaluation(status="ok", score=10.0))
            tree = FlowGraph("min")
            tree.add(root)
            coding = Coding()
            CoordinateAgent(0, config, Ideas(), coding, Experiment(), Memory(), tree).research_round(1, root)
            self.assertEqual(coding.debug_calls, 1)
            self.assertEqual(len(tree.nodes), 3)
            self.assertTrue(any(node.depth == 2 and node.evaluation.score == 8.0
                                for node in tree.nodes.values()))
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
            survey_dir = temp / "ideation_surveys"
            survey_dir.mkdir()
            (survey_dir / "parent.json").write_text('{"focus":"closing edge"}',
                                                    encoding="utf-8")
            checkpoint = scientist.save(f"test_{uuid.uuid4().hex}")
            first = json.loads((temp / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(first["completed_rounds"], 1)
            self.assertEqual(len(first["root_pool"]), 2)
            self.assertTrue((checkpoint / "memory.sqlite3").is_file())
            self.assertTrue((checkpoint / "ideation_agent_state.json").is_file())
            self.assertTrue((checkpoint / "ideation_surveys" / "parent.json").is_file())
            baseline_code = Path(first["solutions"][0]["code_path"])
            baseline_code.unlink()
            (survey_dir / "parent.json").unlink()
            self.assertFalse(baseline_code.exists())
            resumed_config = Config.from_dict(json.loads((checkpoint / "config.json").read_text(encoding="utf-8")))
            resumed_config.rounds = 2
            COPScientist(resumed_config, FakeModel(), FakeSearch(), checkpoint.name).run()
            second = json.loads((temp / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(second["completed_rounds"], 2)
            self.assertTrue(baseline_code.is_file())
            self.assertTrue((survey_dir / "parent.json").is_file())
            self.assertEqual(len(second["solutions"]), 3)
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
                Idea("Route", "Test", "Order", "Shorter"), None, temp / "solution.py")
            self.assertEqual(result.status, "ok")
            self.assertEqual(len(model.users), 2)
            self.assertIn("'(' was never closed", model.users[1])
        finally:
            shutil.rmtree(temp)

    def test_coding_generation_includes_parent_research_context(self):
        class RecordingModel:
            user = ""

            def complete(self, system, user):
                self.user = user
                return "def solve(instance):\n    return []\n"

        class AcceptEvaluator:
            def evaluate(self, path):
                return Evaluation(status="ok", score=1.0)

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            parent = Solution(id="parent", idea=Idea("Baseline", "Measured", "Tour", "Shorter"),
                              code="def solve(instance):\n    return [0]\n",
                              metrics={"gap": 3.0}, features=(3,), score=3.0,
                              summary="Keep the feasible tour")
            model = RecordingModel()
            result = CodingAgent(config, model, AcceptEvaluator()).run(
                Idea("Improve", "Test", "Swap", "Shorter"), parent, temp / "solution.py",
                "Avoid repeated distances")
            self.assertEqual(result.status, "ok")
            self.assertIn('"parent_solution"', model.user)
            self.assertIn('"score": 3.0', model.user)
            self.assertIn('"required_signature": "def solve(instance)"', model.user)
            self.assertIn("Keep the feasible tour", model.user)
            self.assertIn("Avoid repeated distances", model.user)
        finally:
            shutil.rmtree(temp)

    def test_coding_debug_retries_invalid_revision_with_full_output(self):
        class RevisionModel:
            def __init__(self):
                self.users = []

            def complete(self, system, user):
                self.users.append(user)
                if len(self.users) == 1:
                    return "not a diff"
                return "<<<<<<< SEARCH\n    return []\n=======\n    return [0]\n>>>>>>> REPLACE"

        class FailThenPass:
            calls = 0

            def evaluate(self, path):
                self.calls += 1
                if self.calls == 1:
                    return Evaluation(status="failed", feedback="short failure",
                                      output_path=str(raw_output))
                return Evaluation(status="ok", score=1.0)

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.debug_attempts = 2
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            raw_output = temp / "evaluation.txt"
            raw_output.write_text("full evaluator traceback marker", encoding="utf-8")
            path = temp / "solution.py"
            path.write_text("def solve(instance):\n    return []\n", encoding="utf-8")
            model, evaluator = RevisionModel(), FailThenPass()
            result = CodingAgent(config, model, evaluator).debug(
                Idea("Route", "Test", "Order", "Shorter"), None, path)
            self.assertEqual(result.status, "ok")
            self.assertEqual(evaluator.calls, 2)
            self.assertIn("full evaluator traceback marker", model.users[0])
            self.assertIn("Invalid debugging response", model.users[1])
            self.assertIn("return [0]", path.read_text(encoding="utf-8"))
            self.assertEqual(len(list((temp / "details").glob("*_debug_*.txt"))), 2)
        finally:
            shutil.rmtree(temp)

    def test_agent_prompts_are_external_and_renderable(self):
        for stage in ("literature_review", "code_generation",
                      "code_debugging", "experiment_reflection", "experiment_summary",
                      "long_term_reflection"):
            self.assertTrue(load_prompt(f"system_{stage}_copscientist.txt"))
            self.assertIn("example", render_prompt(f"user_{stage}_copscientist.txt", "example"))
        for mode in ("coordinated", "independent"):
            self.assertTrue(load_prompt(f"system_idea_{mode}_generation_copscientist.txt"))
            for suffix in ("", "_solver"):
                user = load_prompt(f"user_idea_{mode}_generation{suffix}_copscientist.txt")
                self.assertIn("example", user.format(
                    problem_description="example", function_to_evolve="solve", obj_type="min",
                    function_description="signature", parent_solutions="parent",
                    current_research_flow_graph="tree", long_term_reflection="reflection",
                    research_context="memory", num_ideas=2))
        self.assertTrue(load_prompt("system_ideation_search_decision_copscientist.txt"))
        self.assertIn("example", render_prompt("user_ideation_search_decision_copscientist.txt",
                                                "example"))
        with self.assertRaises(ValueError):
            load_prompt("../config.yaml")

    def test_search_replace_code_revision(self):
        source = "def solve(instance):\n    return []\n"
        diff = "<<<<<<< SEARCH\n    return []\n=======\n    return [0]\n>>>>>>> REPLACE"
        self.assertIn("return [0]", update_code(source, diff))
        self.assertEqual(parse_code("Explanation first.\n```python\ndef solve(instance):\n    return []\n```"),
                         source)
        spaced = "def solve(instance):\n    return []   \n"
        self.assertIn("return [0]", update_code(spaced, diff))

    def test_experiment_restores_best_version_after_failed_revision(self):
        class BadRevisionModel:
            def __init__(self):
                self.reflections = 0

            def complete(self, system, user):
                if "Examine the latest measured result" in system:
                    self.reflections += 1
                    if self.reflections == 1:
                        diff = ("<<<<<<< SEARCH\n"
                                "    return list(range(len(instance['coordinates'])))\n"
                                "=======\n    return []\n>>>>>>> REPLACE")
                        return experiment_reply("no", "Try a local revision", diff)
                    return experiment_reply(thinking="Revision is infeasible")
                return json.dumps({"diagnosis": "Revision failed", "evidence": "Infeasible route",
                                   "next_step": "Use previous version"})

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.experiment_repeats = 2
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            path = temp / "solution.py"
            original = "def solve(instance):\n    return list(range(len(instance['coordinates'])))\n"
            path.write_text(original, encoding="utf-8")
            agent = ExperimentAgent(config, BadRevisionModel())
            initial = Evaluation(status="ok", score=10.0)
            result, history, _, compact = agent.run(
                Idea("Route", "Test", "Order", "Shorter"), path, initial, 12.0)
            self.assertEqual(result.status, "ok")
            self.assertEqual(len(history), 2)
            self.assertEqual(history[1]["evaluation"]["status"], "failed")
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(compact, [])
        finally:
            shutil.rmtree(temp)

    def test_experiment_reflection_uses_parent_and_full_output(self):
        class RecordingModel:
            def __init__(self):
                self.reflection_prompt = ""

            def complete(self, system, user):
                if "final report" in system.lower():
                    return json.dumps({"summary": "No local change was justified."})
                self.reflection_prompt = user
                return experiment_reply(thinking="No measured gain")

        class FixedEvaluator:
            def evaluate(self, path, *, callbacks=None):
                return Evaluation(status="ok", score=10.0, feedback="short",
                                  output_path=str(output))

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.experiment_repeats = 1
        config.verbose = False
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            code = temp / "solution.py"
            code.write_text("def solve(instance):\n    return []\n", encoding="utf-8")
            output = temp / "eval_output.txt"
            output.write_text("full diagnostic marker", encoding="utf-8")
            parent = Solution(id="parent", idea=Idea("Baseline", "Test", "Tour", "Feasible"),
                              code="def solve(instance):\n    return []\n", score=12.0,
                              summary="Parent experiment insight")
            model = RecordingModel()
            result, history, summary, compact = ExperimentAgent(
                config, model, FixedEvaluator()).run(
                Idea("Child", "Test", "Revise", "Improve"), code,
                Evaluation(status="ok", score=10.0, feedback="short",
                           output_path=str(output)), 12.0, max_repeats=1, parent=parent)
            self.assertEqual(result.score, 10.0)
            self.assertEqual(len(history), 1)
            self.assertEqual(summary, "No local change was justified.")
            self.assertEqual(compact, [])
            self.assertIn("full diagnostic marker", model.reflection_prompt)
            self.assertIn("Parent experiment insight", model.reflection_prompt)
        finally:
            shutil.rmtree(temp)

    def test_experiment_invalid_diff_is_recorded_before_next_reflection(self):
        class InvalidDiffModel:
            def __init__(self):
                self.reflections = 0

            def complete(self, system, user):
                if "final report" in system.lower():
                    return json.dumps({"summary": "The invalid patch was rejected."})
                self.reflections += 1
                if self.reflections == 1:
                    return experiment_reply(
                        "no", "Try patch",
                        "<<<<<<< SEARCH\nmissing\n=======\nreplacement\n>>>>>>> REPLACE")
                return experiment_reply(thinking="Patch did not apply")

        class FixedEvaluator:
            def evaluate(self, path, *, callbacks=None):
                return Evaluation(status="ok", score=10.0)

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.verbose = False
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            code = temp / "solution.py"
            original = "def solve(instance):\n    return []\n"
            code.write_text(original, encoding="utf-8")
            model = InvalidDiffModel()
            best, history, _, compact = ExperimentAgent(
                config, model, FixedEvaluator()).run(
                Idea("Route", "Test", "Order", "Shorter"), code,
                Evaluation(status="ok", score=10.0), 12.0, max_repeats=2)
            self.assertEqual(model.reflections, 2)
            self.assertIn("must match exactly once", history[0]["revision_error"])
            self.assertEqual(best.score, 10.0)
            self.assertEqual(code.read_text(encoding="utf-8"), original)
            self.assertEqual(compact, [])
        finally:
            shutil.rmtree(temp)

    def test_experiment_reevaluates_and_compacts_history(self):
        class ReflectionModel:
            def __init__(self):
                self.prompts = []

            def complete(self, system, user):
                if "final report" in system.lower():
                    return json.dumps({"summary": "Four measurements were summarized."})
                self.prompts.append(user)
                return experiment_reply("no", "Keep measuring")

        class FixedEvaluator:
            calls = 0

            def evaluate(self, path, *, callbacks=None):
                self.calls += 1
                return Evaluation(status="ok", score=float(self.calls))

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.verbose = False
        config.summarization_interval = 2
        config.compact_history_length = 1
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            code = temp / "solution.py"
            code.write_text("def solve(instance):\n    return []\n", encoding="utf-8")
            model, evaluator = ReflectionModel(), FixedEvaluator()
            best, history, summary, compact = ExperimentAgent(
                config, model, evaluator).run(
                    Idea("Route", "Test", "Order", "Shorter"), code,
                    Evaluation(status="ok", score=-100.0), 12.0, max_repeats=4)
            self.assertEqual(evaluator.calls, 4)
            self.assertEqual(best.score, 1.0)
            self.assertEqual(len(compact), 2)
            self.assertEqual(len(history), 1)
            self.assertIn("Four measurements", summary)
            self.assertIn("progressive_summary", model.prompts[2])
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
            return {"message": {"items": [{"title": ["A scheduling paper"], "DOI": "10.1/a",
                                            "published": {"date-parts": [[2020]]}}]}}

        def post(url, payload):
            return {"results": [{"title": "Routing guide", "url": "https://example.org/route"}]}

        search = LiteratureSearch(fetch=fetch, post=post)
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test"}):
            sources, errors = search.review(["scheduling"], ["routing"], 2, True)
        self.assertEqual(errors, [])
        self.assertEqual({s.kind for s in sources}, {"paper", "web"})
        self.assertEqual(sources[0].url, "https://doi.org/10.1/a")

    def test_tavily_search_reads_only_bounded_source_pages(self):
        requests = []
        pages = []

        def post(url, payload):
            requests.append((url, payload))
            return {"results": [{"title": "Algorithm note", "url": "https://example.org/a",
                                 "content": "Search excerpt"},
                                {"title": "Another note", "url": "https://example.org/b",
                                 "content": "Another excerpt"}]}

        def fetch_page(url, max_chars=8000):
            pages.append(url)
            return "Full page explains the algorithm and its assumptions."

        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}):
            search = LiteratureSearch(post=post, fetch_page=fetch_page)
            sources = search.web("routing heuristic", 2)
        self.assertEqual(requests[0][0], "https://api.tavily.com/search")
        self.assertEqual(requests[0][1]["query"], "routing heuristic")
        self.assertEqual(sources[0].content_level, "snippet")
        self.assertEqual(search.read_sources(sources, max_pages=1), [])
        self.assertEqual(pages, ["https://example.org/a"])
        self.assertEqual(sources[0].content_level, "page")
        self.assertEqual(sources[1].content_level, "snippet")

    def test_webpage_reader_extracts_text_and_drops_scripts(self):
        from email.message import Message
        from io import BytesIO
        from copscientist.search import _get_page

        class Response(BytesIO):
            def __init__(self):
                super().__init__(b"<html><h1>Route method</h1><script>ignore me</script>"
                                 b"<p>Try the nearest feasible node.</p></html>")
                self.headers = Message()
                self.headers["Content-Type"] = "text/html; charset=utf-8"

        with patch("copscientist.search.urlopen", return_value=Response()):
            content = _get_page("https://example.org/method", max_chars=100)
        self.assertIn("Route method", content)
        self.assertIn("nearest feasible node", content)
        self.assertNotIn("ignore me", content)

    def test_optional_survey_synthesis_failure_keeps_basic_ideation_available(self):
        class Search:
            def review(self, paper_queries, web_queries, limit, required):
                return [Source(title="Paper", url="https://example.org/paper", kind="paper",
                               query="method", abstract="A method abstract")], []

        class Model:
            def complete(self, system, user):
                return "not JSON"

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.verbose = False
        agent = IdeationAgent(config, Model(), Search())
        sources, errors = agent.prepare_review()
        self.assertEqual(len(sources), 1)
        self.assertEqual(agent.review["claims"], [])
        self.assertIn("Survey synthesis failed", errors[0])

    def test_enabled_survey_is_saved_before_research_rounds(self):
        class Search:
            def review(self, paper_queries, web_queries, limit, required):
                return [Source(title="Relevant paper", url="https://example.org/paper",
                               kind="paper", query="route", abstract="Measured local costs")], []

            def read_sources(self, sources, max_pages=6):
                return []

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.task.instances = ["tsplib_euc_2d/eil51.tsp.gz"]
        config.task.validation_instances = ["tsplib_euc_2d/eil76.tsp.gz"]
        config.rounds = 0
        config.search_enabled = True
        config.verbose = False
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            config.output_dir = temp
            COPScientist(config, FakeModel(), Search()).run()
            survey = json.loads((temp / "literature_review.json").read_text(encoding="utf-8"))
            self.assertEqual(survey["review"]["claims"][0]["source_urls"],
                             ["https://example.org/paper"])
            self.assertEqual(survey["sources"][0]["content_level"], "abstract")
        finally:
            shutil.rmtree(temp)

    def test_survey_refines_one_query_and_keeps_only_readable_citations(self):
        class Search:
            def __init__(self):
                self.queries = []

            def review(self, paper_queries, web_queries, limit, required):
                self.queries.append((paper_queries, web_queries))
                query = (paper_queries or web_queries)[0]
                return [Source(title=query, url=f"https://example.org/{len(self.queries)}",
                               kind="paper" if paper_queries else "web", query=query,
                               abstract="A concrete method" if paper_queries else "Snippet",
                               content_level="abstract" if paper_queries else "snippet")], []

            def read_sources(self, sources, max_pages=6):
                for source in sources[:max_pages]:
                    if source.kind == "web":
                        source.content = "The page explains a concrete solver mechanism."
                        source.content_level = "page"
                return []

        class SurveyModel:
            def __init__(self):
                self.calls = []

            def complete(self, system, user):
                self.calls.append(user)
                if len(self.calls) == 1:
                    return json.dumps({"claims": [{"claim": "Supported", "source_urls":
                                                    ["https://example.org/1"]}],
                                       "transferable_methods": [], "limitations": [],
                                       "sufficient": False,
                                       "follow_up_queries": [{"kind": "web", "query": "specific solver mechanism"},
                                                             {"kind": "web", "query": "extra query"}]})
                return json.dumps({"claims": [{"claim": "Grounded in page", "source_urls":
                                                ["https://example.org/2"]},
                                               {"claim": "Unverified", "source_urls":
                                                ["https://invented.example/none"]}],
                                   "transferable_methods": [], "limitations": [], "sufficient": True,
                                   "follow_up_queries": []})

        config = load_config(PROJECT_ROOT / "config.yaml", "tsp")
        config.verbose = False
        config.search_max_queries = 4
        search, model = Search(), SurveyModel()
        agent = IdeationAgent(config, model, search)
        sources, errors = agent.prepare_review()
        self.assertEqual(errors, [])
        self.assertEqual(len(search.queries), 2)
        self.assertEqual(search.queries[1], ([], ["specific solver mechanism"]))
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(sources), 2)
        self.assertEqual(agent.review["claims"],
                         [{"claim": "Grounded in page", "source_urls": ["https://example.org/2"]}])
        self.assertEqual(len(agent.review["searches"]), 4)

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
            self.assertEqual(len(result["solutions"]), 2)
            self.assertEqual(result["solutions"][1]["parent_id"], result["solutions"][0]["id"])
            self.assertEqual(result["solutions"][1]["evaluation"]["status"], "ok")
            child_path = Path(result["solutions"][1]["code_path"])
            self.assertEqual(child_path.parent.parent.name, "solver")
            self.assertEqual(child_path.name, f"{child_path.parent.name}_solution.py")
            self.assertEqual(model.coding_calls, 2)
            self.assertEqual(result["validation"]["status"], "ok")
            self.assertIsNotNone(best.evaluation.score)
            import sqlite3
            with closing(sqlite3.connect(temp / "memory.sqlite3")) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 0)
        finally:
            shutil.rmtree(temp)

    def test_heuristic_solution_is_isolated_from_seed(self):
        config = load_config(PROJECT_ROOT / "config.yaml", "tsp_constructive")
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            solution = temp / "test_solution.py"
            solution.write_text("def select_next_node(*args, **kwargs):\n    raise RuntimeError('solution marker')\n", encoding="utf-8")
            result = ExperimentAgent(config).evaluate(solution)
            self.assertEqual(result.status, "failed")
            self.assertIn("solution marker", result.feedback)
            self.assertNotIn("solution marker", (config.problem_path / config.task.seed_file).read_text(encoding="utf-8"))
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
            self.assertIsInstance(baseline, Solution)
            self.assertEqual(baseline.evaluation.status, "ok")
            self.assertEqual(baseline.score, baseline.evaluation.score)
            self.assertEqual(baseline.code, Path(baseline.code_path).read_text(encoding="utf-8"))
            baseline_path = Path(baseline.code_path)
            self.assertEqual(baseline_path.parent.parent.name, "heuristic")
            self.assertEqual(baseline_path.name, f"{baseline.id_str()}_solution.py")
            self.assertEqual((baseline_path.parent / f"{baseline.id_str()}_eval.py").read_text(encoding="utf-8"),
                             (config.problem_path / "eval.py").read_text(encoding="utf-8"))
            self.assertEqual(result["validation"]["status"], "ok")
            self.assertEqual(list(baseline.evaluation.metrics), [50])
            self.assertGreater(len(result["validation"]["metrics"]), 0)
            self.assertEqual(len(baseline.evaluation.features), 1)
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
            solutions = result["solutions"]
            self.assertEqual(len(solutions), 3)
            self.assertEqual(solutions[2]["parent_id"], solutions[1]["id"])
            self.assertEqual(len(solutions[1]["intermediate"]), 2)
            round_tree = json.loads((temp / "details" / "coordinate_0_round_1_tree.json").read_text(encoding="utf-8"))
            self.assertEqual(len(round_tree["nodes"]), 2)
            self.assertIn(solutions[1]["id"], result["root_pool"])
            self.assertNotIn(solutions[2]["id"], result["root_pool"])
            initial = solutions[1]["intermediate"][0]["evaluation"]["score"]
            self.assertLess(solutions[1]["evaluation"]["score"], initial)
            self.assertIn("remaining =", Path(solutions[1]["code_path"]).read_text(encoding="utf-8"))
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
