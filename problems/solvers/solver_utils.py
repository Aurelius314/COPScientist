"""Standalone evaluation entry point shared by solver tasks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import runpy
import statistics
import sys
import traceback

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from copscientist.evaluator import load_instance, validate


def _load_solver(source: Path, callbacks_file: Path | None = None):
    spec = importlib.util.spec_from_file_location("evaluated_solver", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import solver: {source}")
    module = importlib.util.module_from_spec(spec)
    if callbacks_file is not None:
        module.Callbacks = runpy.run_path(str(callbacks_file))["Callbacks"]
    spec.loader.exec_module(module)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise ValueError(f"{source} must define solve(instance)")
    return solve


def _reference(kind: str, problem_dir: Path, instance_path: Path) -> float | None:
    if kind == "tsp":
        records = json.loads((problem_dir / "tsplib_euc_2d" / "optimal_values.json")
                             .read_text(encoding="utf-8"))["instances"]
        return records.get(instance_path.name.split(".")[0], {}).get("optimal_length")
    if kind == "cvrp":
        solution_path = problem_dir / "X-solutions" / f"{instance_path.stem}.sol"
        if solution_path.is_file():
            match = re.search(r"^Cost\s+(\d+(?:\.\d+)?)",
                              solution_path.read_text(encoding="utf-8"), re.M)
            return float(match.group(1)) if match else None
    if kind == "jssp":
        records = json.loads((problem_dir / "bks.json").read_text(encoding="utf-8"))
        return next((record["upper_bound"] for record in records
                     if record["instance"] == instance_path.stem), None)
    return None


def _emit(metrics: dict[str, float]) -> None:
    if not metrics or any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("Evaluation produced no finite metrics")
    features = tuple(int(value) for value in metrics.values())
    score = statistics.mean(metrics.values())
    print("__SANDBOX_RESULT__")
    print("__METRICS_START__")
    print(repr(metrics))
    print("__METRICS_END__")
    print("__FEATURES_START__")
    print(repr(features))
    print("__FEATURES_END__")
    print("__SCORE_START__")
    print(repr(score))
    print("__SCORE_END__")
    print("__SANDBOX_SUCCESS__")


def run(problem_dir: Path) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a solver on bundled instances.")
    parser.add_argument("--mode", choices=("train", "val"), default="val")
    parser.add_argument("--solver_file", type=Path, help="Solver module; defaults to seed_solution.py")
    parser.add_argument("--callbacks_file", type=Path, help="Optional Callbacks class for the solver")
    parser.add_argument("--max_instances", type=int, help="Limit instances for a quick check")
    parser.add_argument("--instances", nargs="+", help="Override instance globs from settings.yaml")
    parser.add_argument("--root_dir", type=Path, help="Project root; defaults to this script's project")
    parser.add_argument("--file_output_prefix", default="", help="Reserved for CLI compatibility")
    args = parser.parse_args()
    try:
        problem_dir = problem_dir.resolve()
        if args.root_dir is not None:
            project_root = args.root_dir.resolve()
            problem_dir = project_root / "problems" / "solvers" / problem_dir.name
        settings = yaml.safe_load((problem_dir / "settings.yaml").read_text(encoding="utf-8"))["task"]
        kind = settings["entry_function"]
        patterns = args.instances or (settings.get("validation_instances") if args.mode == "val"
                                      else settings["instances"])
        patterns = patterns or settings["instances"]
        paths = sorted({path for pattern in patterns for path in problem_dir.glob(pattern)
                        if path.is_file()})
        limit = args.max_instances if args.max_instances is not None else settings["max_instances"]
        if limit < 1:
            raise ValueError("max_instances must be positive")
        paths = paths[:limit]
        if not paths:
            raise ValueError(f"No {args.mode} instances found in {problem_dir}")
        source = (args.solver_file or problem_dir / settings["seed_file"]).resolve()
        solve = _load_solver(source, args.callbacks_file)
        metrics = {}
        for path in paths:
            instance = load_instance(kind, path)
            objective = float(validate(kind, instance, solve(instance)))
            if not math.isfinite(objective):
                raise ValueError(f"Nonfinite objective for {path.name}")
            reference = _reference(kind, problem_dir, path)
            if kind != "3d-bpp":
                if reference is None or reference <= 0:
                    raise ValueError(f"Missing positive reference for {path.name}")
                value = 100 * (objective - reference) / reference
            else:
                value = objective
            key = path.relative_to(problem_dir).as_posix()
            metrics[key] = float(value)
            print(f"[*] {key}: objective={objective}, metric={value}")
        print("[*] Metric: " + ("bins" if kind == "3d-bpp" else "gap_percent"))
        _emit(metrics)
    except Exception as exc:
        print("__SANDBOX_ERROR__:")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc
