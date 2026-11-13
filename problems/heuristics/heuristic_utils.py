"""Small helpers shared by the migrated heuristic evaluators."""

import argparse
import inspect
import math



def get_heuristic_name(module, possible_names: list[str]) -> str | None:
    for name in possible_names:
        if hasattr(module, name) and inspect.isfunction(getattr(module, name)):
            return name
    return None


def parse_eval_args(default_problem_size: int, *, with_method: bool = False,
                    aco_options: bool = False):
    """Accept the standalone flags and the workflow's legacy positional arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a heuristic.")
    parser.add_argument("legacy_problem_size", nargs="?", type=int)
    parser.add_argument("legacy_root_dir", nargs="?")
    parser.add_argument("legacy_mode", nargs="?", choices=("train", "val"))
    if with_method:
        parser.add_argument("legacy_method", nargs="?", choices=("sample", "aco"))
        parser.add_argument("--method", choices=("sample", "aco"))
    parser.add_argument("--problem_size", type=int)
    parser.add_argument("--root_dir")
    parser.add_argument("--file_output_prefix", default="")
    parser.add_argument("--mode", choices=("train", "val"))
    if aco_options:
        parser.add_argument("--iterations", type=int, default=100)
        parser.add_argument("--ants", type=int, default=30)
        parser.add_argument("--max_instances", type=int)
    args = parser.parse_args()
    args.problem_size = args.problem_size or args.legacy_problem_size or default_problem_size
    args.root_dir = args.root_dir or args.legacy_root_dir
    args.mode = args.mode or args.legacy_mode or "val"
    if with_method:
        args.method = args.method or args.legacy_method or "aco"
    if aco_options and (args.iterations < 1 or args.ants < 1 or
                        (args.max_instances is not None and args.max_instances < 1)):
        parser.error("iterations, ants, and max_instances must be positive")
    return args

def emit_heuristic_result(metrics: dict[int | str, float]) -> None:
    """Write the shared eval.py protocol after problem-specific tests finish."""
    if not metrics:
        raise ValueError("Evaluation produced no metrics")
    values = {key: float(value) for key, value in metrics.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("Evaluation produced a nonfinite metric")
    features = tuple(int(value) for value in values.values())
    score = sum(values.values()) / len(values)
    print("__SANDBOX_RESULT__")
    print("__METRICS_START__")
    print(repr(values))
    print("__METRICS_END__")
    print("__FEATURES_START__")
    print(repr(features))
    print("__FEATURES_END__")
    print("__SCORE_START__")
    print(repr(score))
    print("__SCORE_END__")
    print("__SANDBOX_SUCCESS__")
