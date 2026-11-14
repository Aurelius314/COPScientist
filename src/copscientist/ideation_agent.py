"""Ideation Agent: OR-Agent-style idea generation for one research-tree parent."""

from __future__ import annotations

import json
import sys
import uuid
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
    def __init__(self, config: Config, model: Model, search: LiteratureSearch,
                 survey_model: Model | None = None):
        # =====General configuration and LLM client=====
        self.config, self.model, self.search = config, model, search
        self.survey_model = survey_model or model
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
        suffix = "_solver" if config.task.kind == "solver" else ""
        self.user_idea_coordinated_generation_prompt = load_prompt(
            f"user_idea_coordinated_generation{suffix}_copscientist.txt")
        self.user_idea_independent_generation_prompt = load_prompt(
            f"user_idea_independent_generation{suffix}_copscientist.txt")
        self.system_ideation_search_decision_prompt = load_prompt(
            "system_ideation_search_decision_copscientist.txt")
        self.user_ideation_search_decision_prompt = load_prompt(
            "user_ideation_search_decision_copscientist.txt")
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
        # EvoScientist-style broad search, source reading, reflection, and one narrow follow-up.
        query_budget = self.config.search_max_queries
        broad_papers = [f"{self.config.task.name} combinatorial optimization algorithm",
                        *self.config.literature_queries][:min(2, query_budget)]
        broad_web = [f"{self.config.task.name} heuristic solver implementation",
                     *self.config.web_queries][:min(1, query_budget - len(broad_papers))]
        searched = [{"kind": "paper", "query": query} for query in broad_papers]
        searched += [{"kind": "web", "query": query} for query in broad_web]
        self.sources, errors = self.search.review(
            broad_papers, broad_web, self.config.search_results,
            self.config.search_required)
        reader = getattr(self.search, "read_sources", None)
        if reader:
            errors.extend(reader(self.sources, max_pages=self.config.search_max_pages))

        def summarize_sources() -> dict:
            readable = {source.url for source in self.sources
                        if source.content or source.content_level == "abstract"}
            if self.config.search_required and not readable:
                raise RuntimeError("Search required but no source text could be read")
            if not self.sources:
                return {"claims": [], "transferable_methods": [],
                        "limitations": errors, "searches": searched}
            payload = [{**asdict(source), "abstract": source.abstract[:800],
                        "content": source.content[:5000]}
                       for source in self.sources[:20]]
            try:
                response = self.survey_model.complete(
                    load_prompt("system_literature_review_copscientist.txt"),
                    render_prompt("user_literature_review_copscientist.txt",
                                  json.dumps({"task": self.config.task.description,
                                              "sources": payload, "searches": searched,
                                              "search_errors": errors},
                                             ensure_ascii=False)))
                self.total_responses += 1
                raw = parse_json(response)
            except (ValueError, RuntimeError, OSError) as exc:
                if self.config.search_required:
                    raise
                errors.append(f"Survey synthesis failed: {exc}")
                return {"claims": [], "transferable_methods": [],
                        "limitations": errors, "searches": searched}
            return {"claims": _grounded(raw.get("claims"), "claim", readable),
                    "transferable_methods": _grounded(raw.get("transferable_methods"),
                                                       "method", readable),
                    "limitations": [*(raw.get("limitations")
                                       if isinstance(raw.get("limitations"), list) else []),
                                    *errors],
                    "follow_up_queries": raw.get("follow_up_queries", []),
                    "sufficient": raw.get("sufficient", True),
                    "searches": searched}

        survey = summarize_sources()
        remaining = query_budget - len(searched)
        if survey.get("sufficient") is False and remaining > 0:
            follow_up = []
            existing = {(item["kind"], item["query"].lower()) for item in searched}
            for item in (survey.get("follow_up_queries")
                         if isinstance(survey.get("follow_up_queries"), list) else []):
                if not isinstance(item, dict):
                    continue
                kind, query = item.get("kind"), item.get("query")
                if kind not in {"paper", "web"} or not isinstance(query, str):
                    continue
                query = query.strip()[:200]
                if not query or (kind, query.lower()) in existing:
                    continue
                follow_up.append({"kind": kind, "query": query})
                existing.add((kind, query.lower()))
                if len(follow_up) >= remaining:
                    break
            if follow_up:
                extra, extra_errors = self.search.review(
                    [item["query"] for item in follow_up if item["kind"] == "paper"],
                    [item["query"] for item in follow_up if item["kind"] == "web"],
                    self.config.search_results, self.config.search_required)
                errors.extend(extra_errors)
                known = {source.url.lower() for source in self.sources}
                new_sources = [source for source in extra if source.url.lower() not in known]
                if reader:
                    pages_left = max(0, self.config.search_max_pages -
                                     min(len(self.sources), self.config.search_max_pages))
                    errors.extend(reader(new_sources, max_pages=pages_left))
                self.sources.extend(new_sources)
                searched.extend(follow_up)
                survey = summarize_sources()
        survey.pop("follow_up_queries", None)
        survey.pop("sufficient", None)
        self.review = survey
        return self.sources, errors

    def research_for_parent(self, parent: Solution | None, memory: list[dict],
                            tree: FlowGraph, long_term_reflection: str
                            ) -> tuple[dict, list[Source]]:
        """Optionally search the current failure mode before one idea-generation call."""
        if not self.config.search_enabled:
            return {}, []
        context = {
            "task": asdict(self.config.task),
            "parent": ({"idea": asdict(parent.idea) if isinstance(parent.idea, Idea)
                        else parent.idea,
                        "score": parent.score, "metrics": parent.metrics,
                        "feedback": parent.evaluation.feedback[-1500:],
                        "summary": (parent.summary or "")[-2500:],
                        "recent_scores": parent.intermediate_scores[-4:],
                        "code": (parent.code or "")[-5000:]}
                       if parent is not None else None),
            "tree": tree.to_dict(), "memory": memory[:6],
            "long_term_reflection": long_term_reflection,
            "global_survey": self.review,
            "max_queries": self.config.search_idea_max_queries,
        }
        try:
            response = self.survey_model.complete(
                self.system_ideation_search_decision_prompt,
                render_prompt("user_ideation_search_decision_copscientist.txt",
                              json.dumps(context, ensure_ascii=False)))
            self.total_responses += 1
            decision = parse_json(response)
        except (ValueError, RuntimeError, OSError) as exc:
            if self.verbose:
                print(f">>>[IdeationAgent] Local search decision unavailable: {exc}")
            return {}, []
        if decision.get("search") is not True:
            if self.verbose:
                print(">>>[IdeationAgent] Current evidence is sufficient; no local search")
            return {}, []

        # The model chooses concrete queries; limits are enforced before any request.
        queries = []
        seen = set()
        for item in decision.get("queries", []) if isinstance(decision.get("queries"), list) else []:
            if not isinstance(item, dict):
                continue
            kind, query = item.get("kind"), item.get("query")
            if kind not in {"web", "paper"} or not isinstance(query, str):
                continue
            query = query.strip()[:200]
            if not query or (kind, query.lower()) in seen:
                continue
            queries.append({"kind": kind, "query": query})
            seen.add((kind, query.lower()))
            if len(queries) == self.config.search_idea_max_queries:
                break
        if not queries:
            return {}, []
        if self.verbose:
            print(f">>>[IdeationAgent] Searching the current failure mode: "
                  f"{decision.get('failure_mode', '(unspecified)')}")
        sources, errors = self.search.review(
            [item["query"] for item in queries if item["kind"] == "paper"],
            [item["query"] for item in queries if item["kind"] == "web"],
            self.config.search_results, self.config.search_required)
        reader = getattr(self.search, "read_sources", None)
        if reader:
            errors.extend(reader(sources, max_pages=self.config.search_idea_max_pages))
        readable = {source.url for source in sources
                    if source.content or source.content_level == "abstract"}
        if self.config.search_required and not readable:
            raise RuntimeError("Local search required but no source text could be read")
        review = {"focus": str(decision.get("failure_mode", "")),
                  "claims": [], "transferable_methods": [],
                  "limitations": list(errors), "searches": queries}
        if readable:
            payload = [{**asdict(source), "abstract": source.abstract[:800],
                        "content": source.content[:5000]} for source in sources[:10]]
            try:
                response = self.survey_model.complete(
                    load_prompt("system_literature_review_copscientist.txt"),
                    render_prompt("user_literature_review_copscientist.txt",
                                  json.dumps({"task": context["task"],
                                              "focus": review["focus"], "sources": payload,
                                              "searches": queries, "search_errors": errors},
                                             ensure_ascii=False)))
                self.total_responses += 1
                summary = parse_json(response)
                review["claims"] = _grounded(summary.get("claims"), "claim", readable)
                review["transferable_methods"] = _grounded(
                    summary.get("transferable_methods"), "method", readable)
                if isinstance(summary.get("limitations"), list):
                    review["limitations"] = [*summary["limitations"], *errors]
            except (ValueError, RuntimeError, OSError) as exc:
                review["limitations"].append(f"Local survey synthesis failed: {exc}")

        # Keep the evidence for this specific parent separate from the startup survey.
        directory = self.config.run_path / "ideation_surveys"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{uuid.uuid4().hex}.json").write_text(
            json.dumps({"parent_id": parent.id if parent else None,
                        "parent_round": parent.round if parent else None,
                        "decision": decision, "review": review,
                        "sources": [asdict(source) for source in sources],
                        "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
        return review, sources

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

        # Search only when this parent has a concrete need; the OR-Agent branches below stay intact.
        local_review, local_sources = self.research_for_parent(
            parent, memory, tree, long_term_reflection)
        sources_by_url = {source.url.lower(): source for source in self.sources}
        for source in local_sources:
            previous = sources_by_url.get(source.url.lower())
            if previous is None or (source.content and not previous.content):
                sources_by_url[source.url.lower()] = source
        sources = list(sources_by_url.values())
        # Keep COP's memory and literature context alongside OR-Agent's problem context.
        research_context = json.dumps({
            "memory": memory[:6], "review": self.review,
            "current_parent_review": local_review,
            "sources": [{"title": source.title, "url": source.url,
                         "kind": source.kind, "year": source.year,
                         "content_level": source.content_level}
                        for source in sources
                        if source.content or source.content_level == "abstract"],
        }, ensure_ascii=False)
        current_research_flow_graph = json.dumps(tree.to_dict(), ensure_ascii=False)
        allowed = {source.url for source in sources
                   if source.content or source.content_level == "abstract"}

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
    agent = IdeationAgent(config, model, LiteratureSearch())
    ideas = agent.run(seed_solution, [], args.num_ideas, tree,
                      "Improve the seed algorithm's measured score.")
    print("\nGenerated ideas:")
    for index, idea in enumerate(ideas):
        print(f"{index + 1}. {idea}")
