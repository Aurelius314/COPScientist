from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import closing
from pathlib import Path

from copscientist.config import PROJECT_ROOT, load_config
from copscientist.core import COPScientist
from copscientist.experiment_agent import ExperimentAgent
from copscientist.search import LiteratureSearch
from copscientist.utils import load_prompt, render_prompt
from copscientist.evaluator import load_instance, validate


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


class WorkflowTests(unittest.TestCase):
    def test_agent_prompts_are_external_and_renderable(self):
        for stage in ("literature_review", "idea_generation", "code_generation",
                      "code_debugging", "experiment_summary"):
            self.assertTrue(load_prompt(f"system_{stage}_copscientist.txt"))
            self.assertIn("example", render_prompt(f"user_{stage}_copscientist.txt", "example"))
        with self.assertRaises(ValueError):
            load_prompt("../config.yaml")

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
        config.debug_attempts = 1
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
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 4)
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
            self.assertNotIn("candidate marker", (config.problem_path / "gpt.py").read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(temp)

    def test_configuration_precedence(self):
        temp = PROJECT_ROOT / "runs" / f"test_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        try:
            global_path = temp / "config.yaml"
            global_path.write_text('problem: tsp_constructive\nrun:\n  rounds: 5\n', encoding="utf-8")
            config = load_config(global_path, "tsp", {"rounds": 7, "output_dir": Path("runs/custom")})
            self.assertEqual(config.task.name, "tsp")
            self.assertEqual(config.task.objective, "min")
            self.assertEqual(config.rounds, 7)
            self.assertEqual(config.run_path, PROJECT_ROOT / "runs/custom")
        finally:
            shutil.rmtree(temp)


if __name__ == "__main__":
    unittest.main()
