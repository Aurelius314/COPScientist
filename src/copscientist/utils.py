"""Shared records, model adapter, and prompt helpers for COPScientist."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen


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


class Model(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class ChatModel:
    def __init__(self, model: str, base_url: str, api_key_env: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env

    def complete(self, system: str, user: str) -> str:
        key = os.environ.get(self.api_key_env)
        if not key or not self.model:
            raise RuntimeError(f"Set {self.api_key_env} and llm.model before running research")
        body = json.dumps({"model": self.model, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], "temperature": 0.4}).encode()
        request = Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urlopen(request, timeout=120) as response:
            data = json.load(response)
        return data["choices"][0]["message"]["content"]


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
