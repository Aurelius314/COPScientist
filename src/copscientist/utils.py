"""Shared records, model adapter, and prompt helpers for COPScientist."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
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
class Candidate:
    id: str
    round: int
    depth: int
    parent_id: str | None
    idea: Idea
    code_path: str
    evaluation: Evaluation
    reflection: str = ""
    intermediate: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
