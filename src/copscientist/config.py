"""Global settings, per-problem settings, and CLI overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import yaml

from .utils import Task


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass
class Config:
    task: Task
    project_root: Path = PROJECT_ROOT
    output_dir: Path = Path("runs")
    rounds: int = 2
    coordinate_agents: int = 1
    ideas_per_round: int = 2
    max_depth: int = 2
    debug_attempts: int = 1
    experiment_repeats: int = 3
    evaluation_repeats: int = 1
    timeout_seconds: int = 300
    literature_queries: list[str] = field(default_factory=list)
    web_queries: list[str] = field(default_factory=list)
    search_results: int = 3
    search_enabled: bool = False
    search_required: bool = False
    model: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    brave_key_env: str = "BRAVE_SEARCH_API_KEY"

    @property
    def problem_path(self) -> Path:
        path = (self.project_root / self.task.problem_dir).resolve()
        if not path.is_relative_to((self.project_root / "problems").resolve()):
            raise ValueError("problem directory must stay inside problems")
        return path

    @property
    def run_path(self) -> Path:
        path = self.output_dir
        return path if path.is_absolute() else self.project_root / path

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["project_root"] = str(self.project_root)
        data["output_dir"] = str(self.output_dir)
        data["problem_path"] = str(self.problem_path)
        return data


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def _global_path(path: Path | None) -> Path:
    if path is not None:
        return path.resolve()
    local = Path.cwd() / "config.yaml"
    return local if local.is_file() else DEFAULT_CONFIG


def _problem_settings(name: str) -> tuple[Path, str]:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("problem must be a single directory name")
    candidates = [(PROJECT_ROOT / "problems" / kind / name / "settings.yaml", kind)
                  for kind in ("heuristics", "solvers")]
    found = [(path, kind) for path, kind in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one settings.yaml for {name!r} under problems/heuristics or problems/solvers")
    return found[0]


def load_config(path: Path | None = None, problem_name: str | None = None,
                overrides: dict[str, Any] | None = None) -> Config:
    """Load global YAML, then problem YAML, then explicit overrides."""
    data = _read_yaml(_global_path(path))
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    name = overrides.pop("problem", None) or problem_name or data.get("problem")
    if not isinstance(name, str):
        raise ValueError("Set problem in config.yaml or pass --problem")
    settings_path, directory_kind = _problem_settings(name)
    task_data = _read_yaml(settings_path)["task"]
    expected_kind = "heuristic" if directory_kind == "heuristics" else "solver"
    if task_data.get("name") != name or task_data.get("kind") != expected_kind:
        raise ValueError(f"Problem identity/kind does not match {settings_path}")
    task = Task(**task_data, problem_dir=str(settings_path.parent.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    run = dict(data.get("run", {}))
    search = dict(data.get("search", {}))
    llm = dict(data.get("llm", {}))
    run_fields = {"output_dir", "rounds", "coordinate_agents", "ideas_per_round",
                  "max_depth", "debug_attempts", "experiment_repeats", "evaluation_repeats",
                  "timeout_seconds"}
    llm_fields = {"model", "base_url", "api_key_env"}
    for key, value in overrides.items():
        if key in run_fields:
            run[key] = value
        elif key in llm_fields:
            llm[key] = value
        elif key == "search_required":
            search["required"] = value
        else:
            raise ValueError(f"Unknown configuration override: {key}")
    config = Config(
        task=task,
        project_root=PROJECT_ROOT,
        output_dir=Path(run.get("output_dir") or f"runs/{task.name}"),
        rounds=int(run.get("rounds", 2)),
        coordinate_agents=int(run.get("coordinate_agents", 1)),
        ideas_per_round=int(run.get("ideas_per_round", 2)),
        max_depth=int(run.get("max_depth", 2)),
        debug_attempts=int(run.get("debug_attempts", 1)),
        experiment_repeats=int(run.get("experiment_repeats", 3)),
        evaluation_repeats=int(run.get("evaluation_repeats", 1)),
        timeout_seconds=int(run.get("timeout_seconds", 300)),
        literature_queries=list(search.get("literature_queries", [])),
        web_queries=list(search.get("web_queries", [])),
        search_results=int(search.get("results_per_query", 3)),
        search_enabled=bool(search.get("enabled", False)),
        search_required=bool(search.get("required", False)),
        model=str(llm.get("model", "")),
        base_url=str(llm.get("base_url", "https://api.openai.com/v1")),
        api_key_env=str(llm.get("api_key_env", "OPENAI_API_KEY")),
        brave_key_env=str(search.get("brave_key_env", "BRAVE_SEARCH_API_KEY")),
    )
    if config.rounds < 0 or config.debug_attempts < 0:
        raise ValueError("rounds and debug_attempts must be nonnegative")
    if any(value <= 0 for value in (config.coordinate_agents, config.ideas_per_round,
                                    config.max_depth, config.experiment_repeats,
                                    config.evaluation_repeats,
                                    config.timeout_seconds, config.search_results)):
        raise ValueError("Agent counts, depth, repeats, timeout, and search results must be positive")
    return config
