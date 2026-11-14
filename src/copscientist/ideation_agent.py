"""Ideation Agent: OR-Agent-style idea generation for one research-tree parent."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copscientist.config import Config, load_config
from copscientist.flow_graph import FlowGraph
from copscientist.search import LiteratureSearch
from copscientist.utils import (Evaluation, Idea, LLMClient, Model, Solution,
                                Source, extract_json, load_prompt, parse_json,
                                render_prompt)


def _grounded(items: object, field: str, allowed: set[str]) -> list[dict]:
    if not isinstance(items, list):
        return []
    return [
        {field: item[field], "source_urls": item["source_urls"]}
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get(field), str)
        and isinstance(item.get("source_urls"), list)
        and item["source_urls"]
        and all(url in allowed for url in item["source_urls"])
    ]


class IdeationAgent:
    def __init__(self, config: Config, model: Model, search: LiteratureSearch):
        # =====General configuration and LLM client=====
        self.config, self.model, self.search = config, model, search
        self.sources: list[Source] = []
        self.review: dict = {}

        # =====Vars updated during agent running=====
        self.total_responses = 0
        self.function_evals = 0
        self.valid_responses = 0

        # =====Ideation Agent settings=====
        self.num_children = config.ideas_per_round
        self.ideas_coordinated_generation_disabled = config.ideas_coordinated_generation_disabled
        self.verbose = config.verbose

        # =====Load problem data and prompts=====
        self.system_idea_coordinated_generation_prompt = load_prompt(
            "system_idea_coordinated_generation_copscientist.txt")
        self.system_idea_independent_generation_prompt = load_prompt(
            "system_idea_independent_generation_copscientist.txt")
        self.user_idea_coordinated_generation_prompt = load_prompt(
            "user_idea_coordinated_generation_copscientist.txt")
        self.user_idea_independent_generation_prompt = load_prompt(
            "user_idea_independent_generation_copscientist.txt")
        self.function_to_evolve = ("solve" if config.task.kind == "solver"
                                   else config.task.entry_function)
        problem_description_path = config.problem_path / "problem_description.txt"
        function_description_path = config.problem_path / "function_description.txt"
        self.problem_description = (problem_description_path.read_text(encoding="utf-8")
                                    if problem_description_path.is_file()
                                    else config.task.description or "(empty)")
        self.function_description = (function_description_path.read_text(encoding="utf-8")
                                     if function_description_path.is_file()
                                     else f"Implement {self.function_to_evolve} with the interface "
                                          "required by this task's evaluator.")
        if self.verbose:
            print("\n>>>[IdeationAgent] Ideation Agent initialized.")

    def reset(self) -> None:
        """Reset the same three counters as OR-Agent's Idea Agent."""
        self.total_responses = 0
        self.function_evals = 0
        self.valid_responses = 0
        if self.verbose:
            print("\n>>>[IdeationAgent] Ideation Agent reset.")

    def state_dict(self) -> dict[str, int]:
        return {"total_responses": self.total_responses,
                "function_evals": self.function_evals,
                "valid_responses": self.valid_responses}

    def load_state(self, state: dict) -> None:
        self.total_responses = int(state["total_responses"])
        self.function_evals = int(state["function_evals"])
        self.valid_responses = int(state["valid_responses"])

    def save(self, checkpoint: str) -> None:
        """Write OR-Agent's idea-agent state file into a COPScientist checkpoint."""
        if not checkpoint or Path(checkpoint).name != checkpoint:
            raise ValueError("Checkpoint must be a single directory name")
        directory = self.config.project_root / "checkpoints" / checkpoint
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "ideation_agent_state.json").write_text(
            json.dumps(self.state_dict(), indent=4), encoding="utf-8")
        if self.verbose:
            print(f"\n>>>[IdeationAgent] Checkpoint saved to: {directory}")

    def load(self, checkpoint: str) -> None:
        """Load the counters saved by :meth:`save`."""
        if not checkpoint or Path(checkpoint).name != checkpoint:
            raise ValueError("Checkpoint must be a single directory name")
        path = self.config.project_root / "checkpoints" / checkpoint / "ideation_agent_state.json"
        self.load_state(json.loads(path.read_text(encoding="utf-8")))
        if self.verbose:
            print(f"\n>>>[IdeationAgent] Checkpoint loaded from: {path.parent}")

    # survey: web search with literature review.
    def prepare_review(self) -> tuple[list[Source], list[str]]:
        self.sources, errors = self.search.review(
            [f"{self.config.task.name} combinatorial optimization algorithm",
             *self.config.literature_queries],
            [f"{self.config.task.name} heuristic solver implementation",
             *self.config.web_queries],
            self.config.search_results, self.config.search_required,
        )
        payload = [{**asdict(source), "abstract": source.abstract[:500]}
                   for source in self.sources[:20]]
        if payload:
            raw_review = parse_json(self.model.complete(
                load_prompt("system_literature_review_copscientist.txt"),
                render_prompt("user_literature_review_copscientist.txt",
                              json.dumps(payload, ensure_ascii=False)),
            ))
            allowed = {source.url for source in self.sources}
            self.review = {
                "claims": _grounded(raw_review.get("claims"), "claim", allowed),
                "transferable_methods": _grounded(raw_review.get("transferable_methods"), "method", allowed),
                "limitations": raw_review.get("limitations", []),
            }
        else:
            self.review = {"claims": [], "transferable_methods": [], "limitations": errors}
        return self.sources, errors

    # idea generation: the same ordered stages as OR-Agent's IdeaAgent.run.
    def run(self, parent: Solution | None, memory: list[dict], count: int,
            tree: FlowGraph, long_term_reflection: str = "") -> list[Idea]:
        num_ideas = count or self.num_children
        if num_ideas < 1:
            raise ValueError("num_ideas must be positive")
        ideas: list[Idea] = []

        # -----Prepare parent solution string-----
        parent_solutions_str = str(parent) if parent is not None else "(empty)"

        # -----Handle long-term reflection-----
        if long_term_reflection:
            print("\n>>>[IdeationAgent] Long-term reflection is used for this generation.")
        else:
            long_term_reflection = "None"
            print("\n>>>[IdeationAgent] Long-term reflection is not used for this generation.")

        # Keep COP's memory and literature context alongside OR-Agent's problem context.
        research_context = json.dumps({
            "memory": memory[:6], "review": self.review,
            "sources": [asdict(source) for source in self.sources],
        }, ensure_ascii=False)
        current_research_flow_graph = json.dumps(tree.to_dict(), ensure_ascii=False)
        allowed = {source.url for source in self.sources}

        if not self.ideas_coordinated_generation_disabled:
            # -----Coordinated generation-----
            print("\n>>>[IdeationAgent] Idea generations are coordinated.")
            user = self.user_idea_coordinated_generation_prompt.format(
                problem_description=self.problem_description,
                function_to_evolve=self.function_to_evolve,
                obj_type=self.config.task.objective,
                function_description=self.function_description,
                current_research_flow_graph=current_research_flow_graph,
                parent_solutions=parent_solutions_str,
                long_term_reflection=long_term_reflection,
                research_context=research_context,
                num_ideas=num_ideas,
            )
            attempt = 0
            while True:
                attempt += 1
                response = self.model.complete(self.system_idea_coordinated_generation_prompt, user)
                self.total_responses += 1
                parsed = extract_json(response)
                try:
                    ideas = [Idea(title=item["title"], rationale=item["rationale"],
                                  algorithm=item["algorithm"],
                                  expected_effect=item["expected_effect"],
                                  sources=[url for url in item.get("sources", []) if url in allowed])
                             for item in parsed] if isinstance(parsed, list) else []
                except (KeyError, TypeError):
                    ideas = []
                if not ideas:
                    print(f"\n>>>[IdeationAgent] Warning: LLM response could not be parsed as ideas "
                          f"(attempt {attempt}):\n{response}.")
                    if attempt >= 5:
                        raise RuntimeError("\n>>>[IdeationAgent] Max attempts reached for generating ideas")
                else:
                    self.valid_responses += 1
                    break
        else:
            # -----Independent generation-----
            print("\n>>>[IdeationAgent] Idea generations are independent.")
            user = self.user_idea_independent_generation_prompt.format(
                problem_description=self.problem_description,
                function_to_evolve=self.function_to_evolve,
                obj_type=self.config.task.objective,
                function_description=self.function_description,
                parent_solutions=parent_solutions_str,
                long_term_reflection=long_term_reflection,
                research_context=research_context,
                num_ideas=1,
            )
            attempt = 0
            while True:
                attempt += 1
                response = self.model.complete(self.system_idea_independent_generation_prompt, user)
                self.total_responses += 1
                parsed = extract_json(response)
                try:
                    item = parsed["idea"] if isinstance(parsed, dict) else None
                    idea = Idea(title=item["title"], rationale=item["rationale"],
                                algorithm=item["algorithm"],
                                expected_effect=item["expected_effect"],
                                sources=[url for url in item.get("sources", []) if url in allowed])
                except (KeyError, TypeError):
                    idea = None
                if idea is None:
                    print(f"\n>>>[IdeationAgent] Warning: LLM response could not be parsed as an idea "
                          f"(attempt {attempt}):\n{response}.")
                    if attempt >= 100:
                        raise RuntimeError("\n>>>[IdeationAgent] Max attempts reached for generating ideas")
                    continue
                ideas.append(idea)
                self.valid_responses += 1
                if len(ideas) == num_ideas:
                    break

        if len(ideas) != num_ideas:
            print(f"\n>>>[IdeationAgent] Warn: len(ideas) = {len(ideas)}, "
                  f"but {num_ideas} required")
        if self.verbose:
            print(f"\n>>>[IdeationAgent] {len(ideas)} ideas generated:")
            for index, idea in enumerate(ideas):
                print(f"{index + 1}. {idea}")
        return ideas

if __name__ == "__main__":
    # For debugging purposes: generate ideas from the configured seed solution.
    import argparse

    parser = argparse.ArgumentParser(description="Test the Ideation Agent on a seed solution.")
    parser.add_argument("--problem", default="tsp_constructive")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--num-ideas", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, args.problem)
    config.ideas_per_round = args.num_ideas
    seed_path = config.problem_path / config.task.seed_file
    seed_solution = Solution(
        idea=Idea("Seed", "Provided baseline", "Existing algorithm", "Reference score"),
        code_path=str(seed_path), code=seed_path.read_text(encoding="utf-8"),
        evaluation=Evaluation(status="pending"))
    tree = FlowGraph(config.task.objective, root=seed_solution)

    if args.offline:
        class OfflineModel:
            def complete(self, system: str, user: str) -> str:
                idea = {"title": "Try a local improvement", "rationale": "Seed baseline",
                        "algorithm": "Refine the next decision using local costs",
                        "expected_effect": "Improve the measured score", "sources": []}
                if "Your response must be a JSON object" in system:
                    return json.dumps({"idea": idea})
                return json.dumps([idea for _ in range(args.num_ideas)])

        model = OfflineModel()
    else:
        model = LLMClient(config, temperature=1.3)
    agent = IdeationAgent(config, model, LiteratureSearch(config.brave_key_env))
    ideas = agent.run(seed_solution, [], args.num_ideas, tree,
                      "Improve the seed algorithm's measured score.")
    print("\nGenerated ideas:")
    for index, idea in enumerate(ideas):
        print(f"{index + 1}. {idea}")
