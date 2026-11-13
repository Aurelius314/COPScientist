"""Experiment decisions and evidence summaries around the evaluator."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .config import Config
from .evaluator import Evaluator
from .utils import Evaluation, Idea, Model, Solution, better, load_prompt, parse_json, render_prompt, update_code


class ExperimentAgent:
    def __init__(self, config: Config, model: Model | None = None,
                 evaluator: Evaluator | None = None,
                 summary_model: Model | None = None):
        self.config = config
        self.model = model
        self.summary_model = summary_model or model
        self.evaluator = evaluator or Evaluator(config)
        knowledge = config.problem_path / "external_knowledge.txt"
        self.external_knowledge = (knowledge.read_text(encoding="utf-8")
                                   if knowledge.is_file() else "")
        callbacks_description = config.problem_path / "callbacks_description.txt"
        self.callbacks_description = (callbacks_description.read_text(encoding="utf-8")
                                      if callbacks_description.is_file() else "")

    def evaluate(self, code_path: Path, *, validation: bool = False,
                 callbacks: str | None = None) -> Evaluation:
        return self.evaluator.evaluate(code_path, validation=validation,
                                       callbacks=callbacks)

    def evaluate_solution(self, solution: Solution, *, validation: bool = False,
                          callbacks: str | None = None) -> Solution:
        self.evaluator.run(solution, callbacks=callbacks, validation=validation)
        return solution

    def reflect(self, idea: Idea, code: str, evaluation: Evaluation,
                parent_score: float | None, history: list[dict],
                long_term_reflection: str = "") -> dict:
        """Choose a measured local revision or terminate this experiment."""
        if self.model is None:
            return {"action": "terminate", "reason": "No model configured"}
        compact = history[-self.config.compact_history_length:] if self.config.compact_history_length else []
        if not self.config.progressive_summarization or len(history) < self.config.summarization_interval:
            compact = history
        previous = history[:-len(compact)] if compact else history
        progressive = (self._progressive_summary(previous, self.config.task.objective)
                       if self.config.progressive_summarization and previous else "")
        context = {"idea": asdict(idea), "code": code[-12000:],
                   "evaluation": asdict(evaluation), "parent_score": parent_score,
                   "objective": self.config.task.objective,
                   "long_term_reflection": long_term_reflection,
                   "external_knowledge": (self.external_knowledge if not self.config.external_knowledge_persistence_disabled
                                          else ""),
                   "callbacks_description": self.callbacks_description,
                   "current_callbacks": evaluation.callbacks,
                   "progressive_summary": progressive,
                   "history": [{"step": step["step"], "score": step["evaluation"]["score"],
                                "status": step["evaluation"]["status"],
                                "action": ({"action": step["action"].get("action"),
                                            "reason": str(step["action"].get("reason", ""))[:300]}
                                           if step["action"] else None)}
                               for step in compact]}
        for name in ("problem_description", "function_description", "evaluation_description"):
            source = self.config.problem_path / f"{name}.txt"
            if source.is_file() and (name != "evaluation_description"
                                     or not self.config.evaluation_description_disabled):
                context[name] = source.read_text(encoding="utf-8")[:8000]
        result = parse_json(self.model.complete(
            load_prompt("system_experiment_reflection_copscientist.txt"),
            render_prompt("user_experiment_reflection_copscientist.txt",
                          json.dumps(context, ensure_ascii=False)),
        ))
        if result.get("action") not in {"terminate", "update_code", "update_callbacks"}:
            return {"action": "terminate", "reason": "No valid revision action"}
        if result["action"] == "update_callbacks" and not self.callbacks_description:
            return {"action": "terminate", "reason": "This problem has no callback API"}
        return result

    @staticmethod
    def _progressive_summary(history: list[dict], objective: str) -> str:
        scores = [step["evaluation"]["score"] for step in history
                  if step["evaluation"]["score"] is not None]
        actions = [{"action": step["action"].get("action"),
                    "reason": str(step["action"].get("reason", ""))[:300]}
                   for step in history if step["action"]]
        return json.dumps({"experiments": len(history), "first_score": scores[0] if scores else None,
                           "last_score": scores[-1] if scores else None,
                           "best_score": (min(scores) if objective == "min" else max(scores)) if scores else None,
                           "recent_actions": actions[-3:]}, ensure_ascii=False)

    def run(self, idea: Idea, code_path: Path, initial: Evaluation,
            parent_score: float | None, long_term_reflection: str = "",
            max_repeats: int | None = None) -> tuple[Evaluation, list[dict], str]:
        """Evaluate, reflect, revise, and restore the best tested code version."""
        current = initial
        best = initial
        best_code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
        current_callbacks = initial.callbacks
        history: list[dict] = []
        final_revision = False
        repeats = max_repeats if max_repeats is not None else self.config.experiment_repeats
        for step_number in range(repeats):
            code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
            snapshot = code_path.with_name(f"experiment_{step_number}.py")
            snapshot.write_text(code, encoding="utf-8")
            record = {"step": step_number, "code_path": str(snapshot),
                      "evaluation": asdict(current), "action": None}
            history.append(record)
            if current.status == "ok" and better(current.score, best.score, self.config.task.objective):
                best, best_code = current, code
            if not code:
                break
            try:
                action = self.reflect(idea, code, current, parent_score, history,
                                      long_term_reflection)
            except (ValueError, OSError) as exc:
                action = {"action": "terminate", "reason": str(exc)}
            record["action"] = action
            if action["action"] not in {"update_code", "update_callbacks"}:
                break
            revision = ((action.get("code_diff") or action.get("code"))
                        if action["action"] == "update_code" else action.get("callbacks"))
            if not isinstance(revision, str) or not revision.strip():
                record["action"] = {"action": "terminate", "reason": "Missing experiment revision"}
                break
            try:
                details = self.config.run_path / "details"
                details.mkdir(parents=True, exist_ok=True)
                label = "codediff" if action["action"] == "update_code" else "callbacks"
                (details / f"{code_path.parent.name}_{label}_{step_number + 1}.txt").write_text(
                    revision, encoding="utf-8")
                if action["action"] == "update_code":
                    code_path.write_text(update_code(code, revision), encoding="utf-8")
                else:
                    current_callbacks = revision
                current = self.evaluator.evaluate(code_path, callbacks=current_callbacks)
                final_revision = step_number + 1 == repeats or bool(action.get("terminate_after_update"))
                if action.get("terminate_after_update"):
                    break
            except (OSError, ValueError, SyntaxError) as exc:
                current = Evaluation(status="failed", feedback=str(exc))
                break
        if final_revision:
            final_code = code_path.read_text(encoding="utf-8")
            final_snapshot = code_path.with_name(f"experiment_{len(history)}.py")
            final_snapshot.write_text(final_code, encoding="utf-8")
            history.append({"step": len(history), "code_path": str(final_snapshot),
                            "evaluation": asdict(current), "action": None})
            if current.status == "ok" and better(current.score, best.score, self.config.task.objective):
                best, best_code = current, final_code
        if best_code and code_path.exists():
            code_path.write_text(best_code, encoding="utf-8")
        summary = self.summarize(idea, best, parent_score, history,
                                 long_term_reflection, best_code)
        return best, history, summary

    def summarize(self, idea: Idea, evaluation: Evaluation,
                  parent_score: float | None, history: list[dict] | None = None,
                  long_term_reflection: str = "", best_code: str = "") -> str:
        if self.summary_model is None:
            return evaluation.feedback[:1200]
        try:
            records = history or []
            original_code = (Path(records[0]["code_path"]).read_text(encoding="utf-8")[:8000]
                             if records else "")
            context = {"idea": asdict(idea), "evaluation": asdict(evaluation),
                       "parent_score": parent_score, "objective": self.config.task.objective,
                       "long_term_reflection": long_term_reflection,
                       "original_code": original_code, "best_code": best_code[:8000],
                       "history": [{"step": step["step"], "evaluation": step["evaluation"],
                                    "action": ({"action": step["action"].get("action"),
                                                "reason": str(step["action"].get("reason", ""))[:500]}
                                               if step["action"] else None)}
                                   for step in records]}
            system = load_prompt("system_experiment_summary_copscientist.txt")
            user = render_prompt("user_experiment_summary_copscientist.txt",
                                 json.dumps(context, ensure_ascii=False))
            for _ in range(5):
                try:
                    data = parse_json(self.summary_model.complete(system, user))
                except (ValueError, TypeError):
                    continue
                if isinstance(data.get("summary"), str) and data["summary"].strip():
                    return data["summary"].strip()
                if any(data.get(key) for key in ("diagnosis", "evidence", "next_step")):
                    return json.dumps({key: str(data.get(key, "")) for key in
                                       ("diagnosis", "evidence", "next_step")}, ensure_ascii=False)
            raise ValueError("No valid experiment summary after five responses")
        except Exception as exc:
            return f"Experiment summary unavailable: {exc}; raw feedback: {evaluation.feedback[:1000]}"
