"""Experiment decisions and evidence summaries around the evaluator."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .evaluator import Evaluator
from .utils import Evaluation, Idea, Model, better, load_prompt, parse_json, render_prompt, update_code


class ExperimentAgent:
    def __init__(self, config: Config, model: Model | None = None,
                 evaluator: Evaluator | None = None):
        self.config = config
        self.model = model
        self.evaluator = evaluator or Evaluator(config)

    def evaluate(self, code_path: Path, *, validation: bool = False) -> Evaluation:
        return self.evaluator.evaluate(code_path, validation=validation)

    def reflect(self, idea: Idea, code: str, evaluation: Evaluation,
                parent_score: float | None, history: list[dict],
                long_term_reflection: str = "") -> dict:
        """Choose a measured local revision or terminate this experiment."""
        if self.model is None:
            return {"action": "terminate", "reason": "No model configured"}
        context = {"idea": asdict(idea), "code": code[-12000:],
                   "evaluation": asdict(evaluation), "parent_score": parent_score,
                   "objective": self.config.task.objective,
                   "long_term_reflection": long_term_reflection,
                   "history": [{"step": step["step"], "score": step["evaluation"]["score"],
                                "status": step["evaluation"]["status"]} for step in history]}
        result = parse_json(self.model.complete(
            load_prompt("system_experiment_reflection_copscientist.txt"),
            render_prompt("user_experiment_reflection_copscientist.txt",
                          json.dumps(context, ensure_ascii=False)),
        ))
        if result.get("action") not in {"terminate", "update_code"}:
            return {"action": "terminate", "reason": "No valid revision action"}
        return result

    def run(self, idea: Idea, code_path: Path, initial: Evaluation,
            parent_score: float | None, long_term_reflection: str = "") -> tuple[Evaluation, list[dict], str]:
        """Evaluate, reflect, revise, and restore the best tested code version."""
        current = initial
        best = initial
        best_code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
        history: list[dict] = []
        for step_number in range(self.config.experiment_repeats):
            code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
            snapshot = code_path.with_name(f"experiment_{step_number}.py")
            snapshot.write_text(code, encoding="utf-8")
            record = {"step": step_number, "code_path": str(snapshot),
                      "evaluation": asdict(current), "action": None}
            history.append(record)
            if current.status == "ok" and better(current.score, best.score, self.config.task.objective):
                best, best_code = current, code
            if step_number + 1 >= self.config.experiment_repeats or not code:
                break
            try:
                action = self.reflect(idea, code, current, parent_score, history,
                                      long_term_reflection)
            except (ValueError, OSError) as exc:
                action = {"action": "terminate", "reason": str(exc)}
            record["action"] = action
            if action["action"] != "update_code":
                break
            revision = action.get("code_diff") or action.get("code")
            if not isinstance(revision, str) or not revision.strip():
                record["action"] = {"action": "terminate", "reason": "Missing code revision"}
                break
            try:
                code_path.write_text(update_code(code, revision), encoding="utf-8")
                current = self.evaluator.evaluate(code_path)
            except (OSError, ValueError, SyntaxError) as exc:
                current = Evaluation(status="failed", feedback=str(exc))
                break
        if best_code and code_path.exists():
            code_path.write_text(best_code, encoding="utf-8")
        summary = self.summarize(idea, best, parent_score, history,
                                 long_term_reflection)
        return best, history, summary

    def summarize(self, idea: Idea, evaluation: Evaluation,
                  parent_score: float | None, history: list[dict] | None = None,
                  long_term_reflection: str = "") -> str:
        if self.model is None:
            return evaluation.feedback[:1200]
        try:
            data = parse_json(self.model.complete(
                load_prompt("system_experiment_summary_copscientist.txt"),
                render_prompt(
                    "user_experiment_summary_copscientist.txt",
                    json.dumps({"idea": idea.__dict__, "evaluation": evaluation.__dict__,
                                "parent_score": parent_score,
                            "objective": self.config.task.objective,
                            "long_term_reflection": long_term_reflection,
                            "history": [{"step": step["step"],
                                         "evaluation": step["evaluation"],
                                         "action": step["action"]}
                                        for step in (history or [])]}, ensure_ascii=False)[:12000],
                ),
            ))
            return json.dumps({key: str(data.get(key, "")) for key in
                               ("diagnosis", "evidence", "next_step")}, ensure_ascii=False)
        except Exception as exc:
            return f"Experiment summary unavailable: {exc}; raw feedback: {evaluation.feedback[:1000]}"
