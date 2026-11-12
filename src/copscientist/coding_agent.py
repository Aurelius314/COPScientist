"""Translate an idea into executable candidate code and repair failures."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .utils import Candidate, Idea, Model, load_prompt, parse_code, render_prompt


class CodingAgent:
    def __init__(self, config: Config, model: Model):
        self.config, self.model = config, model

    def implement(self, idea: Idea, parent: Candidate | None, path: Path,
                  feedback: str = "") -> Path:
        parent_code = Path(parent.code_path).read_text(encoding="utf-8") if parent else ""
        problem = self.config.problem_path
        eval_code = ((problem / "eval.py").read_text(encoding="utf-8")[:8000]
                     if self.config.task.kind == "heuristic" else "")
        function_name = "solve" if self.config.task.kind == "solver" else self.config.task.entry_function
        context = {"idea": asdict(idea), "parent_code": parent_code[-10000:],
                   "evaluation_code": eval_code, "feedback": feedback,
                   "entry_function": function_name,
                   "solver_kind": self.config.task.entry_function if self.config.task.kind == "solver" else None,
                   "task_description": self.config.task.description}
        prompt = ("system_code_debugging_copscientist.txt" if feedback
                  else "system_code_generation_copscientist.txt")
        user_prompt = ("user_code_debugging_copscientist.txt" if feedback
                       else "user_code_generation_copscientist.txt")
        response = self.model.complete(
            load_prompt(prompt),
            render_prompt(user_prompt, json.dumps(context, ensure_ascii=False)),
        )
        code = parse_code(response)
        tree = ast.parse(code)
        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == function_name for node in tree.body):
            raise ValueError(f"Generated code lacks {function_name}()")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        return path
