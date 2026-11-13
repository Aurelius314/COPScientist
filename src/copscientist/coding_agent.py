"""Translate an idea into executable candidate code and repair failures."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .evaluator import Evaluator
from .utils import Solution, Evaluation, Idea, Model, load_prompt, parse_code, render_prompt, update_code


class CodingAgent:
    def __init__(self, config: Config, model: Model, evaluator: Evaluator | None = None):
        self.config, self.model = config, model
        self.evaluator = evaluator or Evaluator(config)

    def implement(self, idea: Idea, parent: Solution | None, path: Path,
                  feedback: str = "", long_term_reflection: str = "") -> Path:
        parent_code = Path(parent.code_path).read_text(encoding="utf-8") if parent else ""
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
        context = {"idea": asdict(idea), "parent_code": parent_code[-10000:],
                   "evaluation_code": eval_code, "feedback": feedback,
                   "entry_function": function_name,
                   "solver_kind": self.config.task.entry_function if self.config.task.kind == "solver" else None,
                   "task_description": self.config.task.description,
                   **descriptions}
        context["long_term_reflection"] = long_term_reflection
        response = self.model.complete(
            load_prompt("system_code_generation_copscientist.txt"),
            render_prompt("user_code_generation_copscientist.txt",
                          json.dumps(context, ensure_ascii=False)),
        )
        code = parse_code(response)
        tree = ast.parse(code)
        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == function_name for node in tree.body):
            raise ValueError(f"Generated code lacks {function_name}()")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        return path

    def debug(self, idea: Idea, parent: Solution | None, path: Path,
              feedback: str, long_term_reflection: str = "") -> Evaluation:
        """Revise the failed implementation, then evaluate the new version."""
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if not current:
            self.implement(idea, parent, path, feedback, long_term_reflection)
            return self.evaluator.evaluate(path)
        context = {
            "idea": asdict(idea), "code": current, "evaluation_feedback": feedback,
            "entry_function": "solve" if self.config.task.kind == "solver"
                              else self.config.task.entry_function,
            "task_description": self.config.task.description,
            "long_term_reflection": long_term_reflection,
        }
        problem = self.config.problem_path
        for name in ("problem_description", "function_description", "evaluation_description"):
            source = problem / f"{name}.txt"
            if source.is_file() and (name != "evaluation_description"
                                     or not self.config.evaluation_description_disabled):
                context[name] = source.read_text(encoding="utf-8")[:8000]
        if self.config.task.kind == "heuristic" and not self.config.evaluation_description_disabled:
            context["evaluation_code"] = (problem / "eval.py").read_text(encoding="utf-8")[:8000]
        response = self.model.complete(
            load_prompt("system_code_debugging_copscientist.txt"),
            render_prompt("user_code_debugging_copscientist.txt",
                          json.dumps(context, ensure_ascii=False)),
        )
        revised = update_code(current, response)
        details = self.config.run_path / "details"
        details.mkdir(parents=True, exist_ok=True)
        (details / f"{path.parent.name}_debug_{len(list(details.glob(path.parent.name + '_debug_*.txt'))) + 1}.txt").write_text(
            response, encoding="utf-8")
        tree = ast.parse(revised)
        name = context["entry_function"]
        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == name for node in tree.body):
            raise ValueError(f"Generated code lacks {name}()")
        path.write_text(revised, encoding="utf-8")
        return self.evaluator.evaluate(path)

    def run(self, idea: Idea, parent: Solution | None, path: Path,
            long_term_reflection: str = "") -> Evaluation:
        """OR-Agent sequence: generate, evaluate, then debug while invalid."""
        evaluation = Evaluation(status="failed", feedback="Code generation failed")
        for attempt in range(5):
            try:
                self.implement(idea, parent, path,
                               feedback=evaluation.feedback if attempt else "",
                               long_term_reflection=long_term_reflection)
                evaluation = self.evaluator.evaluate(path)
                break
            except (OSError, ValueError, SyntaxError) as exc:
                evaluation = Evaluation(status="failed", feedback=str(exc))
        for _ in range(self.config.debug_attempts):
            if evaluation.status == "ok":
                break
            try:
                evaluation = self.debug(idea, parent, path, evaluation.feedback,
                                        long_term_reflection)
            except (OSError, ValueError, SyntaxError) as exc:
                evaluation = Evaluation(status="failed", feedback=str(exc))
        return evaluation
