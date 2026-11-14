"""Global settings, per-problem settings, and CLI overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
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
    rounds: int = 10000
    coordinate_agents: int = 1
    ideas_per_round: int = 2
    max_depth: int | None = 3
    debug_attempts: int = 3
    experiment_repeats: int = 3
    evaluation_repeats: int = 1
    max_evolutions: int = 500
    verbose: bool = True
    autosave_interval_minutes: float = 180
    dynamic_experiment_allocation: bool = True
    min_experiment_repeats: int = 1
    max_experiment_repeats_cap: int = 12
    improvement_threshold: float = 0.05
    elitist_experiment_factor: float = 2.0
    progressive_summarization: bool = True
    summarization_interval: int = 4
    compact_history_length: int = 2
    reflection_period: int = 0
    reflection_clearance_period: int | None = None
    reflection_compression: int | None = None
    external_knowledge_persistence_disabled: bool = False
    evaluation_description_disabled: bool = False
    ideas_coordinated_generation_disabled: bool = False
    timeout_seconds: int = 300
    python_path: str | dict[str, str] | None = None
    eval_env_vars: dict[str, str] = field(default_factory=dict)
    literature_queries: list[str] = field(default_factory=list)
    web_queries: list[str] = field(default_factory=list)
    search_results: int = 3
    search_enabled: bool = False
    search_required: bool = False
    evolution_enabled: bool = False
    model: str = ""
    llm_provider: str = "openai"
    model_max_retries: int = 10
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    search_max_queries: int = 5
    search_max_pages: int = 6
    role_models: dict[str, dict[str, Any]] = field(default_factory=dict)

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Reconstruct the exact resolved configuration saved with a checkpoint."""
        values = {item.name: data[item.name] for item in fields(cls) if item.name in data}
        values["task"] = Task(**values["task"])
        values["project_root"] = Path(values["project_root"])
        values["output_dir"] = Path(values["output_dir"])
        return cls(**values)


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
    settings_paths = [(PROJECT_ROOT / "problems" / kind / name / "settings.yaml", kind)
                      for kind in ("heuristics", "solvers")]
    found = [(path, kind) for path, kind in settings_paths if path.is_file()]
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
    evolution = dict(data.get("evolution", {}))
    llm = dict(data.get("llm", {}))
    run_fields = {"output_dir", "rounds", "coordinate_agents", "ideas_per_round",
                  "max_depth", "debug_attempts", "experiment_repeats", "evaluation_repeats",
                  "timeout_seconds", "max_evolutions", "verbose", "autosave_interval_minutes",
                  "dynamic_experiment_allocation", "min_experiment_repeats",
                  "max_experiment_repeats_cap", "improvement_threshold",
                  "elitist_experiment_factor", "progressive_summarization",
                  "summarization_interval", "compact_history_length", "reflection_period",
                  "reflection_clearance_period", "reflection_compression",
                  "external_knowledge_persistence_disabled", "evaluation_description_disabled",
                  "ideas_coordinated_generation_disabled", "python_path", "eval_env_vars"}
    llm_fields = {"model", "llm_provider", "base_url", "api_key_env"}
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
        rounds=int(run.get("rounds", 10000)),
        coordinate_agents=int(run.get("coordinate_agents", 1)),
        ideas_per_round=int(run.get("ideas_per_round", 2)),
        max_depth=(None if run.get("max_depth", 3) is None
                   else int(run.get("max_depth", 3))),
        debug_attempts=int(run.get("debug_attempts", 3)),
        experiment_repeats=int(run.get("experiment_repeats", 3)),
        evaluation_repeats=int(run.get("evaluation_repeats", 1)),
        max_evolutions=int(run.get("max_evolutions", 500)),
        verbose=bool(run.get("verbose", True)),
        autosave_interval_minutes=float(run.get("autosave_interval_minutes", 180)),
        dynamic_experiment_allocation=bool(run.get("dynamic_experiment_allocation", True)),
        min_experiment_repeats=int(run.get("min_experiment_repeats", 1)),
        max_experiment_repeats_cap=int(run.get("max_experiment_repeats_cap", 12)),
        improvement_threshold=float(run.get("improvement_threshold", 0.05)),
        elitist_experiment_factor=float(run.get("elitist_experiment_factor", 2)),
        progressive_summarization=bool(run.get("progressive_summarization", True)),
        summarization_interval=int(run.get("summarization_interval", 4)),
        compact_history_length=int(run.get("compact_history_length", 2)),
        reflection_period=int(run.get("reflection_period", 0)),
        reflection_clearance_period=(int(run["reflection_clearance_period"])
                                     if run.get("reflection_clearance_period") is not None else None),
        reflection_compression=(int(run["reflection_compression"])
                                if run.get("reflection_compression") is not None else None),
        external_knowledge_persistence_disabled=bool(run.get("external_knowledge_persistence_disabled", False)),
        evaluation_description_disabled=bool(run.get("evaluation_description_disabled", False)),
        ideas_coordinated_generation_disabled=bool(run.get("ideas_coordinated_generation_disabled", False)),
        timeout_seconds=int(run.get("timeout_seconds", 300)),
        python_path=run.get("python_path"),
        eval_env_vars={str(key): str(value) for key, value in
                       dict(run.get("eval_env_vars") or {}).items()},
        literature_queries=list(search.get("literature_queries", [])),
        web_queries=list(search.get("web_queries", [])),
        search_results=int(search.get("results_per_query", 3)),
        search_enabled=bool(search.get("enabled", False)),
        search_required=bool(search.get("required", False)),
        evolution_enabled=bool(evolution.get("enabled", False)),
        model=str(llm.get("model", "")),
        llm_provider=str(llm.get("llm_provider", "openai")),
        model_max_retries=int(llm.get("max_retries", 10)),
        base_url=str(llm.get("base_url", "https://api.openai.com/v1")),
        api_key_env=str(llm.get("api_key_env", "OPENAI_API_KEY")),
        search_max_queries=int(search.get("max_queries", 5)),
        search_max_pages=int(search.get("max_pages", 6)),
        role_models=dict(llm.get("roles") or {}),
    )
    if config.rounds < 0 or config.debug_attempts < 0:
        raise ValueError("rounds and debug_attempts must be nonnegative")
    if (config.max_evolutions < 0 or config.autosave_interval_minutes < 0
            or config.compact_history_length < 0 or config.reflection_period < 0):
        raise ValueError("Evolution budget, autosave interval, and compact history length must be nonnegative")
    if config.summarization_interval <= 0:
        raise ValueError("Summarization interval must be positive")
    if config.reflection_clearance_period is not None and config.reflection_clearance_period <= 0:
        raise ValueError("Reflection clearance period must be positive")
    if config.reflection_compression is not None and config.reflection_compression <= 0:
        raise ValueError("Reflection compression must be positive")
    if config.model_max_retries <= 0:
        raise ValueError("Model retries must be positive")
    if any(value <= 0 for value in (config.coordinate_agents, config.ideas_per_round,
                                    config.experiment_repeats,
                                    config.evaluation_repeats,
                                    config.timeout_seconds, config.search_results,
                                    config.search_max_queries, config.search_max_pages,
                                    config.min_experiment_repeats,
                                    config.max_experiment_repeats_cap,
                                    config.summarization_interval)):
        raise ValueError("Agent counts, depth, repeats, timeout, and search results must be positive")
    if config.max_depth is not None and config.max_depth <= 0:
        raise ValueError("Tree depth must be positive or null")
    if config.min_experiment_repeats > config.max_experiment_repeats_cap:
        raise ValueError("Minimum experiment repeats exceed the cap")
    if config.improvement_threshold <= 0 or config.elitist_experiment_factor <= 0:
        raise ValueError("Improvement threshold and elitist experiment factor must be positive")
    return config
