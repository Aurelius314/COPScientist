"""Translate an idea into executable solution code and repair failures."""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copscientist.config import Config, load_config
from copscientist.evaluator import Evaluator
from copscientist.utils import (Solution, Evaluation, Idea, Model, load_prompt, parse_code,
                                render_prompt, solution_code_path, update_code)


class CodingAgent:
    def __init__(self, config: Config, model: Model, evaluator: Evaluator | None = None):
        self.config, self.model = config, model
        self.evaluator = evaluator or Evaluator(config)

    def _parent_context(self, parent: Solution | None) -> dict | None:
        """Give code generation the parent's idea and measured outcome as in OR-Agent."""
        if parent is None:
            return None
        code = parent.code
        if code is None and parent.code_path and Path(parent.code_path).is_file():
            code = Path(parent.code_path).read_text(encoding="utf-8")
        idea = asdict(parent.idea) if isinstance(parent.idea, Idea) else parent.idea
        return {"idea": idea, "code": code, "metrics": parent.metrics or parent.evaluation.metrics,
                "features": parent.features or parent.evaluation.features,
                "score": parent.score if parent.score is not None else parent.evaluation.score,
                "summary": parent.summary or parent.reflection}

    @staticmethod
    def _raw_feedback(evaluation: Evaluation) -> str:
        """Prefer the saved evaluator output over its shortened status message."""
        if evaluation.output_path and Path(evaluation.output_path).is_file():
            return Path(evaluation.output_path).read_text(encoding="utf-8")[-12000:]
        return evaluation.feedback

    def _required_signature(self) -> str:
        name = "solve" if self.config.task.kind == "solver" else self.config.task.entry_function
        seed_file = self.config.task.seed_file
        if seed_file:
            seed = self.config.problem_path / seed_file
            if seed.is_file():
                for node in ast.parse(seed.read_text(encoding="utf-8")).body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                        result = f"def {name}({ast.unparse(node.args)})"
                        if node.returns is not None:
                            result += f" -> {ast.unparse(node.returns)}"
                        return result
        return f"def {name}(instance)" if self.config.task.kind == "solver" else name

    def implement(self, idea: Idea, parent: Solution | None, path: Path,
                  feedback: str = "", long_term_reflection: str = "") -> Path:
        """Generate one complete module, check its entry function, and save it."""
        # Assemble the task contract, parent result, and any previous generation error.
        problem = self.config.problem_path
        eval_code = ((problem / "eval.py").read_text(encoding="utf-8")[:8000]
                     if self.config.task.kind == "heuristic"
                     and not self.config.evaluation_description_disabled else "")
        descriptions = {}
        for name in ("problem_description", "function_description", "evaluation_description"):
            source = problem / f"{name}.txt"
            if source.is_file() and (name != "evaluation_description"
                                     or not self.config.evaluation_description_disabled):
                descriptions[name] = source.read_text(encoding="utf-8")[:8000]
        function_name = "solve" if self.config.task.kind == "solver" else self.config.task.entry_function
        context = {"idea": asdict(idea), "parent_solution": self._parent_context(parent),
                   "evaluation_code": eval_code, "feedback": feedback,
                   "entry_function": function_name,
                   "required_signature": self._required_signature(),
                   "objective": self.config.task.objective,
                   "solver_kind": self.config.task.entry_function if self.config.task.kind == "solver" else None,
                   "task_description": self.config.task.description,
                   **descriptions}
        context["long_term_reflection"] = long_term_reflection
        response = self.model.complete(
            load_prompt("system_code_generation_copscientist.txt"),
            render_prompt("user_code_generation_copscientist.txt",
                          json.dumps(context, ensure_ascii=False)),
        )
        # A syntactically valid module is useful only if the evaluator can call its entry function.
        code = parse_code(response)
        tree = ast.parse(code)
        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == function_name for node in tree.body):
            raise ValueError(f"Generated code lacks {function_name}()")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        if self.config.verbose:
            print(f">>>[CodingAgent] Generated code saved: {path}")
        return path

    def debug(self, idea: Idea, parent: Solution | None, path: Path,
              feedback: Evaluation | str | None = None,
              long_term_reflection: str = "") -> Evaluation:
        """Evaluate if needed, then repair and re-evaluate up to the debug budget."""
        if not path.is_file():
            raise FileNotFoundError(path)
        # run() passes its failed evaluation; a direct debug() call evaluates here first.
        if isinstance(feedback, Evaluation):
            evaluation = feedback
        elif feedback is None:
            if self.config.verbose:
                print(f">>>[CodingAgent] Evaluating existing code before debugging: {path}")
            evaluation = self.evaluator.evaluate(path)
        else:
            evaluation = Evaluation(status="failed", feedback=feedback)
        if evaluation.status == "ok":
            if self.config.verbose:
                print(f">>>[CodingAgent] No debugging needed: score={evaluation.score}")
            return evaluation

        problem = self.config.problem_path
        descriptions = {}
        # Include optional problem descriptions in every repair prompt.
        for name in ("problem_description", "function_description", "evaluation_description"):
            source = problem / f"{name}.txt"
            if source.is_file() and (name != "evaluation_description"
                                     or not self.config.evaluation_description_disabled):
                descriptions[name] = source.read_text(encoding="utf-8")[:8000]
        if self.config.task.kind == "heuristic" and not self.config.evaluation_description_disabled:
            source = problem / "eval.py"
            if source.is_file():
                descriptions["evaluation_code"] = source.read_text(encoding="utf-8")[:8000]
        entry_function = ("solve" if self.config.task.kind == "solver"
                          else self.config.task.entry_function)
        details = self.config.run_path / "details"

        for debug_round in range(1, self.config.debug_attempts + 1):
            if self.config.verbose:
                print(f">>>[CodingAgent] Debugging round {debug_round}/"
                      f"{self.config.debug_attempts} for {path.name}")
            current = path.read_text(encoding="utf-8")
            raw_output = self._raw_feedback(evaluation)

            context = {"idea": asdict(idea), "code": current,
                       "raw_output": raw_output,
                       "entry_function": entry_function,
                       "required_signature": self._required_signature(),
                       "objective": self.config.task.objective,
                       "task_description": self.config.task.description,
                       "long_term_reflection": long_term_reflection,
                       **descriptions}

            # Construct debug prompt
            response = self.model.complete(
                load_prompt("system_code_debugging_copscientist.txt"),
                render_prompt("user_code_debugging_copscientist.txt",
                              json.dumps(context, ensure_ascii=False)),
            )
            details.mkdir(parents=True, exist_ok=True)
            number = len(list(details.glob(path.parent.name + "_debug_*.txt"))) + 1
            (details / f"{path.parent.name}_debug_{number}.txt").write_text(
                response, encoding="utf-8")
            try:
                # Validate the revision before replacing the current source file.
                revised = update_code(current, response)
                tree = ast.parse(revised)
                if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and node.name == entry_function for node in tree.body):
                    raise ValueError(f"Generated code lacks {entry_function}()")
                path.write_text(revised, encoding="utf-8")
                evaluation = self.evaluator.evaluate(path)
                if self.config.verbose:
                    print(f">>>[CodingAgent] Debug evaluation: {evaluation.status}, "
                          f"score={evaluation.score}")
            except (OSError, ValueError, SyntaxError) as exc:
                if self.config.verbose:
                    print(f">>>[CodingAgent] Revision rejected: {exc}")
                evaluation = Evaluation(status="failed", feedback=(
                    f"{raw_output}\nInvalid debugging response: {exc}"))
            if evaluation.status == "ok":
                break
        return evaluation

    def run(self, idea: Idea, parent: Solution | None, path: Path,
            long_term_reflection: str = "") -> Evaluation:
        """OR-Agent sequence: generate, evaluate, then debug while invalid."""
        if self.config.verbose:
            print(f"\n>>>[CodingAgent] Starting solution {path.name} for {self.config.task.name}")
        evaluation = Evaluation(status="failed", feedback="Code generation failed")
        # Retry generation only when the response cannot produce a usable source file.
        for attempt in range(5):
            if self.config.verbose:
                print(f">>>[CodingAgent] Generation attempt {attempt + 1}/5")
            try:
                self.implement(idea, parent, path,
                               feedback=evaluation.feedback if attempt else "",
                               long_term_reflection=long_term_reflection)
                break
            except (OSError, ValueError, SyntaxError) as exc:
                evaluation = Evaluation(status="failed", feedback=str(exc))
                if self.config.verbose:
                    print(f">>>[CodingAgent] Generation failed: {exc}")
        else:
            if self.config.verbose:
                print(">>>[CodingAgent] Stopped after five unsuccessful generation attempts")
            return evaluation
        # Once code exists, use the evaluator result to decide whether debugging is needed.
        try:
            evaluation = self.evaluator.evaluate(path)
        except (OSError, ValueError, SyntaxError) as exc:
            evaluation = Evaluation(status="failed", feedback=str(exc))
        if self.config.verbose:
            print(f">>>[CodingAgent] Initial evaluation: {evaluation.status}, "
                  f"score={evaluation.score}")
        if evaluation.status != "ok":
            try:
                evaluation = self.debug(idea, parent, path, evaluation,
                                        long_term_reflection)
            except (OSError, ValueError, SyntaxError) as exc:
                evaluation = Evaluation(status="failed", feedback=str(exc))
        if self.config.verbose:
            print(f">>>[CodingAgent] Finished {path.name}: {evaluation.status}, "
                  f"score={evaluation.score}")
        return evaluation

if __name__ == "__main__":
    # Debug one complete Coding Agent cycle without starting a research round.
    import argparse
    import uuid

    parser = argparse.ArgumentParser(description="Evaluate a seed, then generate and debug one child solution.")
    parser.add_argument("--problem", default="tsp_constructive")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--idea", default="Improve the seed algorithm while preserving its interface")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--offline", action="store_true",
                        help="Use a deterministic failing implementation and repair it without an LLM")
    args = parser.parse_args()

    config = load_config(args.config, args.problem)
    config.output_dir = (args.output_dir or config.project_root / "runs" / "coding_agent_test"
                         / config.task.name)
    if not config.task.seed_file:
        parser.error(f"{config.task.name} has no seed_file")
    seed_path = config.problem_path / config.task.seed_file
    seed_code = seed_path.read_text(encoding="utf-8")

    if args.offline:
        entry = "solve" if config.task.kind == "solver" else config.task.entry_function
        broken_code = (f"def {entry}(*args, **kwargs):\n"
                       "    raise RuntimeError('offline coding-agent test')\n")

        class OfflineModel:
            def complete(self, system: str, user: str) -> str:
                if "debugging" in system.lower():
                    return ("<<<<<<< SEARCH\n" + broken_code.rstrip() + "\n=======\n"
                            + seed_code.rstrip() + "\n>>>>>>> REPLACE")
                return broken_code

        model: Model = OfflineModel()
    else:
        from copscientist.utils import LLMClient

        role = config.role_models.get("coding", {})
        model = LLMClient(
            config=config,
            llm_provider=str(role.get("llm_provider", config.llm_provider)),
            model_name=str(role.get("model", config.model)),
            base_url=role.get("base_url", config.base_url),
            api_key_env=role.get("api_key_env", config.api_key_env),
            temperature=float(role.get("temperature", 0.4)),
            max_retries=int(role.get("max_retries", config.model_max_retries)),
        )

    evaluator = Evaluator(config)
    parent = Solution(id=f"seed_{uuid.uuid4().hex[:8]}", round=0,
                      coordinate_agent_id=0, solution_count=0,
                      idea=Idea("Seed", "Provided baseline", "Existing implementation", "Reference score"),
                      code=seed_code)
    evaluator.run(parent)
    print(f">>>[CodingAgent test] Seed: {parent.evaluation.status}, score={parent.score}")
    if parent.evaluation.status != "ok":
        raise SystemExit(f"Seed evaluation failed: {parent.evaluation.feedback}")

    child = Solution(id=f"child_{uuid.uuid4().hex[:8]}", round=1,
                     coordinate_agent_id=0, solution_count=1,
                     parent_id=parent.id,
                     idea=Idea("Improve seed", args.idea, args.idea, "Improve the measured score"))
    child_path = solution_code_path(config.run_path, config.task.kind, child.id_str())
    child.evaluation = CodingAgent(config, model, evaluator).run(child.idea, parent, child_path)
    if child_path.is_file():
        child.code_path = str(child_path)
        child.code = child_path.read_text(encoding="utf-8")
    print(f">>>[CodingAgent test] Child: {child.evaluation.status}, score={child.score}")
    print(f">>>[CodingAgent test] Code: {child.code_path}")
    print(f">>>[CodingAgent test] Evaluation output: {child.evaluation.output_path}")
    if child.evaluation.status != "ok":
        raise SystemExit(1)
