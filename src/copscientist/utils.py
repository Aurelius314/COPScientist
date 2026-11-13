"""Shared records, model adapter, and prompt helpers for COPScientist."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


@dataclass
class Source:
    title: str
    url: str
    kind: str
    query: str
    year: int | None = None
    abstract: str = ""


@dataclass
class Idea:
    title: str
    rationale: str
    algorithm: str
    expected_effect: str
    sources: list[str] = field(default_factory=list)


@dataclass
class Evaluation:
    status: str
    score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    feedback: str = ""
    output_path: str | None = None
    callbacks: str | None = None
    features: tuple[int, ...] | None = None


@dataclass
class Solution:
    """One research solution, including its code, evaluation, and experiment history.

    ``evaluation`` and ``intermediate`` retain the existing COPScientist storage
    format. The direct fields follow OR-Agent's Solution interface.
    """

    _next_id: ClassVar[int] = 0

    id: str | int | None = None
    round: int = 0
    depth: int = 0
    parent_id: str | None = None
    idea: Idea | str | None = None
    code_path: str | None = None
    evaluation: Evaluation = field(default_factory=lambda: Evaluation(status="pending"))
    reflection: str = ""
    intermediate: list[dict[str, Any]] = field(default_factory=list)
    coordinate_agent_id: int | None = None
    research_round: int | None = None
    solution_count: int | None = None
    code: str | None = None
    code_filepath: str | None = None
    output: str | None = None
    output_filepath: str | None = None
    metrics: dict[Any, Any] = field(default_factory=dict)
    features: tuple[int, ...] | None = None
    score: float | None = None
    summary: str | None = None
    intermediate_codes: list[str] = field(default_factory=list)
    intermediate_outputs: list[str] = field(default_factory=list)
    intermediate_metrics: list[dict] = field(default_factory=list)
    intermediate_features: list[tuple[int, ...] | None] = field(default_factory=list)
    intermediate_scores: list[float | None] = field(default_factory=list)
    intermediate_actions: list[Any] = field(default_factory=list)
    compact_summaries: list[str] = field(default_factory=list)
    _sync_ready: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = Solution._next_id
            Solution._next_id += 1
        elif isinstance(self.id, int):
            Solution._next_id = max(Solution._next_id, self.id + 1)
        if self.research_round is None:
            self.research_round = self.round
        if self.code_filepath is None:
            self.code_filepath = self.code_path
        if self.code_path is None:
            self.code_path = self.code_filepath
        if self.code is None and self.code_path and Path(self.code_path).is_file():
            self.code = Path(self.code_path).read_text(encoding="utf-8")
        if self.output_filepath is None:
            self.output_filepath = self.evaluation.output_path
        if self.output is None:
            if self.output_filepath and Path(self.output_filepath).is_file():
                self.output = Path(self.output_filepath).read_text(encoding="utf-8")
            else:
                self.output = self.evaluation.feedback
        if not self.metrics:
            self.metrics = dict(self.evaluation.metrics)
        if self.features is None and self.evaluation.features is not None:
            self.features = tuple(self.evaluation.features)
        elif self.features is not None:
            self.features = tuple(self.features)
            self.evaluation.features = self.features
        if self.score is None:
            self.score = self.evaluation.score
        else:
            self.evaluation.score = self.score
            if self.evaluation.status == "pending":
                self.evaluation.status = "ok"
        if self.metrics and not self.evaluation.metrics:
            self.evaluation.metrics = dict(self.metrics)
        if self.summary is None:
            self.summary = self.reflection or None
        elif not self.reflection:
            self.reflection = self.summary
        if self.intermediate and not self.intermediate_scores:
            for record in self.intermediate:
                result = record.get("evaluation", {})
                snapshot = record.get("code_path")
                self.intermediate_codes.append(
                    Path(snapshot).read_text(encoding="utf-8")
                    if snapshot and Path(snapshot).is_file() else "")
                self.intermediate_outputs.append(result.get("feedback", ""))
                self.intermediate_metrics.append(result.get("metrics", {}))
                feature = result.get("features")
                self.intermediate_features.append(tuple(feature) if feature is not None else None)
                self.intermediate_scores.append(result.get("score"))
                self.intermediate_actions.append(record.get("action"))
        self._sync_ready = True

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if not self.__dict__.get("_sync_ready", False):
            return
        if name == "evaluation":
            object.__setattr__(self, "score", value.score)
            object.__setattr__(self, "metrics", dict(value.metrics))
            object.__setattr__(self, "features", value.features)
            object.__setattr__(self, "output_filepath", value.output_path)
            object.__setattr__(self, "output", (
                Path(value.output_path).read_text(encoding="utf-8")
                if value.output_path and Path(value.output_path).is_file()
                else value.feedback))
        elif name in {"score", "metrics", "features"}:
            setattr(self.evaluation, name, value)
            if name == "score" and value is not None and self.evaluation.status == "pending":
                self.evaluation.status = "ok"
        elif name == "output_filepath":
            self.evaluation.output_path = value
        elif name == "output":
            self.evaluation.feedback = (value or "")[-3000:]
        elif name in {"code_path", "code_filepath"}:
            object.__setattr__(self, "code_filepath" if name == "code_path" else "code_path", value)
        elif name == "reflection":
            object.__setattr__(self, "summary", value)
        elif name == "summary":
            object.__setattr__(self, "reflection", value or "")

    def id_str(self, algorithm: str = "copscientist") -> str:
        if algorithm == "copscientist":
            return (f"coordinate{self.coordinate_agent_id}_round{self.research_round}"
                    f"_count{self.solution_count}_id{self.id}")
        return str(self.id)

    def performance_str(self) -> str:
        return f"**Metrics**:\n{self.metrics}\n\n**Score**:\n{self.score}"

    def performance_summary_str(self) -> str:
        return f"{self.performance_str()}\n\n**Summary**:\n{self.summary}"

    def idea_performance_summary_str(self) -> str:
        return f"**Idea**:\n{self.idea}\n\n{self.performance_summary_str()}"

    def experiment_history(self, include_compact_summaries: bool = True,
                           summarization_interval: int = 4) -> str:
        parts = []
        if include_compact_summaries:
            parts.extend(f"Compact Summary Batch #{i + 1}:\n{summary}"
                         for i, summary in enumerate(self.compact_summaries))
        offset = len(self.compact_summaries) * summarization_interval
        for index, action in enumerate(self.intermediate_actions):
            parts.append(
                f"Experiment #{offset + index + 1}:\n"
                f"**Stdout**:\n{self.intermediate_outputs[index]}\n\n"
                f"**Metrics**:\n{self.intermediate_metrics[index]}\n\n"
                f"**Features**:\n{self.intermediate_features[index]}\n\n"
                f"**Score**: {self.intermediate_scores[index]}\n\n"
                f"**Reflections and actions**:\n{action}")
        return "\n\n".join(parts)

    def __str__(self) -> str:
        return (f"**ID**\n{self.id}\n\n**Idea**:\n{self.idea}\n\n"
                f"**Code**:\n```python\n{self.code or ''}\n```\n\n"
                f"**Features**:\n{self.features}\n\n"
                f"{self.performance_summary_str()}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_sync_ready", None)
        return data


@dataclass
class Task:
    name: str
    kind: str
    problem_dir: str
    objective: str
    entry_function: str
    seed_file: str | None = None
    train_args: list[str] = field(default_factory=list)
    validation_args: list[str] = field(default_factory=list)
    instances: list[str] = field(default_factory=list)
    validation_instances: list[str] = field(default_factory=list)
    max_instances: int = 3
    aggregate: str = "mean"
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"heuristic", "solver"}:
            raise ValueError("task kind must be heuristic or solver")
        if self.objective not in {"min", "max"}:
            raise ValueError("objective must be min or max")
        if self.aggregate not in {"mean", "last"}:
            raise ValueError("aggregate must be mean or last")


def better(left: float | None, right: float | None, objective: str) -> bool:
    if left is None:
        return False
    if right is None:
        return True
    return left < right if objective == "min" else left > right


def experiment_allocation(parent_score: float | None, child_score: float | None,
                          objective: str, base: int, minimum: int, cap: int,
                          threshold: float, elitist_factor: float,
                          is_elitist: bool) -> int:
    """OR-Agent's score-dependent experiment budget without population fields."""
    if parent_score is None or child_score is None:
        allocated = base
    else:
        improvement = (child_score - parent_score if objective == "max"
                       else parent_score - child_score)
        relative = improvement / abs(parent_score) if parent_score else improvement
        if relative > threshold:
            allocated = min(int(base * (1 + relative / threshold)), cap)
        elif relative > 0:
            allocated = base
        else:
            allocated = minimum
    if is_elitist:
        allocated = int(allocated * elitist_factor)
    return max(minimum, min(allocated, cap))


class Model(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class ChatModel:
    def __init__(self, model: str, base_url: str, api_key_env: str,
                 temperature: float = 0.4, max_retries: int = 10):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_retries = max_retries

    def complete(self, system: str, user: str) -> str:
        key = os.environ.get(self.api_key_env)
        if not key or not self.model:
            raise RuntimeError(f"Set {self.api_key_env} and llm.model before running research")
        body = json.dumps({"model": self.model, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], "temperature": self.temperature}).encode()
        request = Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        for attempt in range(self.max_retries):
            try:
                with urlopen(request, timeout=120) as response:
                    data = json.load(response)
                return data["choices"][0]["message"]["content"]
            except HTTPError as exc:
                if exc.code < 500 and exc.code != 429:
                    raise
                if attempt + 1 == self.max_retries:
                    raise
            except (URLError, TimeoutError):
                if attempt + 1 == self.max_retries:
                    raise
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Model returned no response")


@dataclass
class Usage:
    """Count completed model responses and candidate evaluations by role."""
    responses: dict[str, int] = field(default_factory=dict)
    evaluations: dict[str, int] = field(default_factory=dict)

    @property
    def total_responses(self) -> int:
        return sum(self.responses.values())

    @property
    def total_evaluations(self) -> int:
        return sum(self.evaluations.values())


class CountingModel:
    def __init__(self, model: Model, usage: Usage, role: str):
        self.model, self.usage, self.role = model, usage, role

    def complete(self, system: str, user: str) -> str:
        response = self.model.complete(system, user)
        self.usage.responses[self.role] = self.usage.responses.get(self.role, 0) + 1
        return response


class CountingEvaluator:
    def __init__(self, evaluator: Any, usage: Usage, role: str):
        self.evaluator, self.usage, self.role = evaluator, usage, role

    def evaluate(self, code_path: Path, *, validation: bool = False,
                 callbacks: str | None = None) -> Evaluation:
        result = self.evaluator.evaluate(code_path, validation=validation, callbacks=callbacks)
        self.usage.evaluations[self.role] = self.usage.evaluations.get(self.role, 0) + 1
        return result


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def parse_code(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    compile(text, "candidate.py", "exec")
    return text + "\n"


def update_code(code: str, response: str) -> str:
    """Apply OR-Agent-style SEARCH/REPLACE blocks or a complete Python module."""
    marker = "<<<<<<< SEARCH\n"
    if marker not in response:
        return parse_code(response)
    updated = code
    blocks = response.split(marker)[1:]
    if not blocks:
        raise ValueError("No SEARCH/REPLACE blocks found")
    for block in blocks:
        if "\n=======\n" not in block or "\n>>>>>>> REPLACE" not in block:
            raise ValueError("Malformed SEARCH/REPLACE block")
        old, remainder = block.split("\n=======\n", 1)
        new = remainder.split("\n>>>>>>> REPLACE", 1)[0]
        if not old or updated.count(old) != 1:
            raise ValueError("SEARCH text must match exactly once")
        updated = updated.replace(old, new, 1)
    compile(updated, "candidate.py", "exec")
    return updated.rstrip() + "\n"


PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(name: str) -> str:
    if not name or Path(name).name != name or not name.endswith(".txt"):
        raise ValueError(f"Invalid prompt name: {name!r}")
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def render_prompt(name: str, context: str) -> str:
    template = load_prompt(name)
    marker = "{{context}}"
    if template.count(marker) != 1:
        raise ValueError(f"Prompt {name} must contain one {marker} marker")
    return template.replace(marker, context)
