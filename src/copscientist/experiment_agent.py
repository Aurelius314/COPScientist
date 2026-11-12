"""Experiment decisions and evidence summaries around the evaluator."""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .evaluator import Evaluator
from .utils import Evaluation, Idea, Model, load_prompt, parse_json, render_prompt


class ExperimentAgent:
    def __init__(self, config: Config, model: Model | None = None,
                 evaluator: Evaluator | None = None):
        self.config = config
        self.model = model
        self.evaluator = evaluator or Evaluator(config)

    def evaluate(self, code_path: Path, *, validation: bool = False) -> Evaluation:
        return self.evaluator.evaluate(code_path, validation=validation)

    def summarize(self, idea: Idea, evaluation: Evaluation,
                  parent_score: float | None) -> str:
        if self.model is None:
            return evaluation.feedback[:1200]
        try:
            data = parse_json(self.model.complete(
                load_prompt("system_experiment_summary_copscientist.txt"),
                render_prompt(
                    "user_experiment_summary_copscientist.txt",
                    json.dumps({"idea": idea.__dict__, "evaluation": evaluation.__dict__,
                                "parent_score": parent_score,
                                "objective": self.config.task.objective}, ensure_ascii=False)[:12000],
                ),
            ))
            return json.dumps({key: str(data.get(key, "")) for key in
                               ("diagnosis", "evidence", "next_step")}, ensure_ascii=False)
        except Exception as exc:
            return f"Experiment summary unavailable: {exc}; raw feedback: {evaluation.feedback[:1000]}"
