"""
# Evaluator
Evaluator class is used for evaluating solution functions.

Execute solution algorithms and return objective values plus diagnostics.
"""

from __future__ import annotations

import ast
import json
import gzip
import getpass
import importlib.util
import math
import os
import re
import runpy
import statistics
import subprocess
import sys
import uuid
import socket
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copscientist.config import Config, load_config
from copscientist.utils import Evaluation, Solution, solution_code_path


def _parse_result_value(source: str):
    """Parse result literals, including NumPy scalar reprs, without executing code."""
    def convert(node):
        if isinstance(node, ast.Constant):
            if type(node.value) not in (str, int, float, bool, type(None)):
                raise ValueError("Unsupported literal in evaluator result")
            if type(node.value) is float and not math.isfinite(node.value):
                raise ValueError("Nonfinite metric in evaluator result")
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(convert(item) for item in node.elts)
        if isinstance(node, ast.List):
            return [convert(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {convert(key): convert(value) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = convert(node.operand)
            if type(value) not in (int, float):
                raise ValueError("Unary operator requires a number")
            return value if isinstance(node.op, ast.UAdd) else -value
        if (isinstance(node, ast.Call) and not node.keywords and len(node.args) == 1
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"np", "numpy"}
                and node.func.attr in {"float16", "float32", "float64", "int8", "int16",
                                       "int32", "int64", "uint8", "uint16", "uint32", "uint64"}):
            value = convert(node.args[0])
            if type(value) not in (int, float):
                raise ValueError("NumPy scalar representation requires a number")
            return (float(value) if node.func.attr.startswith("float") else int(value))
        raise ValueError("Unsupported expression in evaluator result")

    return convert(ast.parse(source, mode="eval").body)


def _parse_eval_result(stdout: str) -> tuple[dict | None, tuple[int, ...] | None, float]:
    """Read the OR-Agent result protocol emitted by a problem's eval.py."""
    lines = [line.strip() for line in stdout.splitlines()]
    if lines.count("__SANDBOX_RESULT__") != 1 or lines.count("__SANDBOX_SUCCESS__") != 1:
        raise ValueError("eval.py did not report a successful structured result")
    result_start = lines.index("__SANDBOX_RESULT__")
    result_end = lines.index("__SANDBOX_SUCCESS__")
    if result_start >= result_end:
        raise ValueError("eval.py result markers are out of order")

    def field(name: str):
        start, end = f"__{name}_START__", f"__{name}_END__"
        if lines.count(start) != 1 or lines.count(end) != 1:
            raise ValueError(f"eval.py must report exactly one {name} block")
        left, right = lines.index(start), lines.index(end)
        if not result_start < left < right < result_end or right != left + 2:
            raise ValueError(f"Malformed {name} block")
        return _parse_result_value(lines[left + 1])

    metrics = field("METRICS")
    features = field("FEATURES")
    score = field("SCORE")
    if metrics is not None and not isinstance(metrics, dict):
        raise ValueError("eval.py metrics must be a dict or None")
    if features is not None and (not isinstance(features, tuple) or any(
            type(value) is not int for value in features)):
        raise ValueError("eval.py features must be an integer tuple or None")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("eval.py score must be finite")
    return metrics, features, float(score)


def _average_metrics(records: list[dict | None]) -> dict:
    """Average matching numeric metrics while retaining detailed run data."""
    if len(records) == 1:
        return records[0] or {}

    def combine(values):
        if all(type(value) in (int, float) for value in values):
            return statistics.mean(values)
        if all(isinstance(value, dict) for value in values):
            keys = list(values[0])
            if all(list(value) == keys for value in values):
                return {key: combine([value[key] for value in values]) for key in keys}
        if all(value == values[0] for value in values):
            return values[0]
        raise ValueError("Metrics have different structures")

    if any(record is None for record in records):
        return {"runs": records}
    try:
        return combine(records)
    except ValueError:
        return {"runs": records}


class Evaluator:
    """Evaluate a Solution through its problem's tests and preserve raw output."""

    def __init__(self, config: Config):
        self.config = config
        self.problem_dir = config.problem_path
        self.output_dir = config.run_path
        self.function_to_evolve = config.task.entry_function
        self.obj_type = config.task.objective
        self.timeout_seconds = config.timeout_seconds
        self.verbose = config.verbose
        eval_path = self.problem_dir / "eval.py"
        self.eval_code = eval_path.read_text(encoding="utf-8") if eval_path.is_file() else None
        default_callbacks = self.problem_dir / "default_callbacks.py"
        self.default_callbacks = (default_callbacks.read_text(encoding="utf-8")
                                  if default_callbacks.is_file() else None)

    def run(self, solution: Solution, callbacks: str | None = None,
            *, validation: bool = False) -> tuple[str, dict | None, tuple[int, ...] | None, float | None]:
        """Evaluate Solution.code and return OR-Agent's raw output/result tuple."""
        code = solution.code
        source_path = Path(solution.code_path) if solution.code_path else None
        if code is None and source_path is not None and source_path.is_file():
            code = source_path.read_text(encoding="utf-8")
        if code is None:
            raise ValueError("Solution has no code to evaluate")

        solution_root = (self.output_dir / "solutions").resolve()
        source_resolved = source_path.resolve() if source_path is not None else None
        correct_kind = (source_resolved is not None and
                        (source_resolved.is_relative_to(solution_root / self.config.task.kind)
                         or source_resolved.parent.parent == solution_root))
        use_existing = (source_path is not None and source_path.is_file()
                        and correct_kind
                        and source_path.read_text(encoding="utf-8") == code)
        if use_existing:
            code_path = source_path
        else:
            label = re.sub(r"[^A-Za-z0-9_-]", "_", solution.id_str())[:70]
            code_path = solution_code_path(self.output_dir, self.config.task.kind, label)
            if code_path.exists():
                code_path = solution_code_path(
                    self.output_dir, self.config.task.kind, f"{label}_{uuid.uuid4().hex[:8]}")
            code_path.parent.mkdir(parents=True, exist_ok=False)
            code_path.write_text(code, encoding="utf-8")
        solution.code = code
        solution.code_path = str(code_path)
        evaluation = self.evaluate(code_path, validation=validation, callbacks=callbacks)
        solution.evaluation = evaluation
        raw_output = solution.output or evaluation.feedback
        if evaluation.status != "ok":
            return raw_output, None, None, None
        return raw_output, evaluation.metrics, evaluation.features, evaluation.score

    def evaluate_individual(self, solution: Solution) -> Solution:
        """Evaluate one solution and return the updated record."""
        if solution.code is None and not (solution.code_path and Path(solution.code_path).is_file()):
            solution.evaluation = Evaluation(status="failed", feedback="Solution has no code to evaluate")
            return solution
        self.run(solution)
        return solution

    @staticmethod
    def _save_output(code_path: Path, output: str) -> str:
        prefix = code_path.stem.removesuffix("_solution")
        target = code_path.parent / f"{prefix}_stdout_{uuid.uuid4().hex[:10]}.txt"
        target.write_text(output, encoding="utf-8")
        return str(target)

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        source = str(self.config.project_root / "src")
        env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
        env.update(self.config.eval_env_vars)
        return env

    def _run(self, command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
        return subprocess.run(command, cwd=cwd, env=self._env(), capture_output=True,
                              text=True, timeout=timeout, check=False)

    def _python(self) -> str:
        configured = self.config.python_path
        if isinstance(configured, dict):
            user = getpass.getuser()
            return (configured.get(f"{user}@{socket.gethostname()}")
                    or configured.get(user) or sys.executable)
        return configured or sys.executable

    def _callback_path(self, code_path: Path, callbacks: str | None) -> str:
        callbacks = callbacks or self.default_callbacks
        if callbacks is None:
            return "-"
        tree = ast.parse(callbacks)
        if not any(isinstance(node, ast.ClassDef) and node.name == "Callbacks"
                   for node in tree.body):
            raise ValueError("Callback code must define class Callbacks")
        prefix = code_path.stem.removesuffix("_solution")
        target = code_path.parent / f"{prefix}_callbacks_{uuid.uuid4().hex[:10]}.py"
        target.write_text(callbacks, encoding="utf-8")
        return str(target)

    def evaluate(self, code_path: Path, *, validation: bool = False,
                 callbacks: str | None = None) -> Evaluation:
        active_callbacks = callbacks or self.default_callbacks
        try:
            callback_path = self._callback_path(code_path, active_callbacks)
            result = self._evaluate_problem(code_path, validation, callback_path)
            result.callbacks = active_callbacks
            return result
        except subprocess.TimeoutExpired as exc:
            def as_text(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

            output = f"Evaluation timed out after {exc.timeout} seconds\n"
            output += as_text(exc.stdout) + "\n" + as_text(exc.stderr)
            return Evaluation(status="failed", feedback=output[-3000:],
                              output_path=self._save_output(code_path, output),
                              callbacks=active_callbacks)
        except (OSError, ValueError, SyntaxError) as exc:
            return Evaluation(status="failed", feedback=str(exc)[:3000], callbacks=active_callbacks)

    def _evaluate_problem(self, code_path: Path, validation: bool, callback_path: str) -> Evaluation:
        if self.eval_code is None:
            raise ValueError(f"Problem has no eval.py: {self.problem_dir}")
        if self.config.task.kind == "heuristic":
            args = list(self.config.task.validation_args if validation else self.config.task.train_args)
            if not args:
                raise ValueError("heuristic task requires train_args / validation_args")
        else:
            args = ["--mode", "val" if validation else "train", "--solver_file", str(code_path),
                    "--max_instances", str(self.config.task.max_instances)]
            patterns = (self.config.task.validation_instances if validation
                        and self.config.task.validation_instances else self.config.task.instances)
            if patterns:
                args.extend(["--instances", *patterns])
            if callback_path != "-":
                args.extend(["--callbacks_file", callback_path])
        prefix = code_path.stem.removesuffix("_solution")
        eval_snapshot = code_path.parent / f"{prefix}_eval.py"
        eval_snapshot.write_text(self.eval_code, encoding="utf-8")
        if "parse_eval_args(" in self.eval_code or "--root_dir" in self.eval_code:
            if "--root_dir" not in args:
                args.extend(["--root_dir", str(self.config.project_root)])
            if "--file_output_prefix" not in args:
                args.extend(["--file_output_prefix", str(code_path.parent / f"{prefix}_file_")])
        results = []
        outputs = []
        for _ in range(self.config.evaluation_repeats):
            result = self._run(
                [self._python(), "-u", "-m", "copscientist.evaluator", "problem",
                 self.config.task.kind, str(code_path), str(self.config.problem_path),
                 str(eval_snapshot), callback_path, *args],
                self.config.problem_path, self.config.timeout_seconds,
            )
            if result.returncode:
                output = "\n".join(outputs + [result.stdout, result.stderr])
                return Evaluation(status="failed", feedback=(result.stderr or result.stdout)[-3000:],
                                  output_path=self._save_output(code_path, output))
            try:
                results.append(_parse_eval_result(result.stdout))
            except (ValueError, SyntaxError) as exc:
                output = "\n".join(outputs + [result.stdout, result.stderr])
                detail = (result.stderr or result.stdout)[-2500:]
                return Evaluation(status="failed", feedback=f"{exc}: {detail}"[-3000:],
                                  output_path=self._save_output(code_path, output))
            outputs.append(result.stdout)
        metrics = _average_metrics([result[0] for result in results])
        features = (results[0][1] if all(result[1] == results[0][1] for result in results)
                    else None)
        return Evaluation(status="ok", score=statistics.mean(result[2] for result in results),
                          metrics=metrics, features=features,
                          feedback="\n".join(outputs)[-3000:],
                          output_path=self._save_output(code_path, "\n".join(outputs)))

def _lines(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return stream.read().splitlines()


def _tsplib(path: Path) -> dict:
    lines = _lines(path)
    if not any(line.replace(" ", "").upper() == "EDGE_WEIGHT_TYPE:EUC_2D" for line in lines):
        raise ValueError("Only TSPLIB EUC_2D instances are supported")
    coords = {}
    section = False
    for line in lines:
        stripped = line.strip()
        if stripped == "NODE_COORD_SECTION":
            section = True
            continue
        if section:
            if stripped in {"EOF", "DISPLAY_DATA_SECTION"}:
                break
            parts = stripped.split()
            if len(parts) >= 3:
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
    if not coords:
        raise ValueError("TSP instance requires NODE_COORD_SECTION")
    return {"name": path.name.split(".")[0], "coordinates": [coords[i] for i in sorted(coords)]}


def _cvrp(path: Path) -> dict:
    capacity = None
    coords, demands = {}, {}
    section = ""
    for line in _lines(path):
        stripped = line.strip()
        if stripped.startswith("CAPACITY"):
            capacity = int(stripped.replace(":", " ").split()[-1])
        elif stripped in {"NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION", "EOF"}:
            section = stripped
        elif section == "NODE_COORD_SECTION" and stripped:
            parts = stripped.split()
            coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif section == "DEMAND_SECTION" and stripped:
            parts = stripped.split()
            demands[int(parts[0])] = int(parts[1])
    if capacity is None or set(coords) != set(demands) or 1 not in coords:
        raise ValueError("Malformed CVRP instance")
    return {"name": path.stem, "capacity": capacity, "coordinates": coords,
            "demands": demands, "depot": 1}


def load_instance(kind: str, path: Path) -> dict:
    if kind == "tsp":
        return _tsplib(path)
    if kind == "cvrp":
        return _cvrp(path)
    if kind == "jssp":
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"name": data["instance"], "jobs": data["jobs"],
                "machines": data["machines"], "operations": data["operations"]}
    if kind == "3d-bpp":
        rows = [[int(x) for x in line.split()] for line in _lines(path) if line.strip()]
        n, width, height, depth = rows[0]
        if len(rows) != n + 1:
            raise ValueError("Malformed 3D-BPP instance")
        return {"name": path.stem, "bin_size": [width, height, depth], "items": rows[1:]}
    raise ValueError(f"Unknown solver kind: {kind}")


def _distance(a, b) -> int:
    return int(math.floor(math.hypot(a[0] - b[0], a[1] - b[1]) + 0.5))


def _permutation(values, n) -> bool:
    return isinstance(values, list) and len(values) == n and all(type(x) is int for x in values) and set(values) == set(range(n))


def validate(kind: str, instance: dict, answer) -> float:
    if kind == "tsp":
        nodes = instance["coordinates"]
        if not _permutation(answer, len(nodes)):
            raise ValueError("TSP answer must be a permutation of zero-based node indices")
        return float(sum(_distance(nodes[answer[i]], nodes[answer[(i + 1) % len(answer)]])
                         for i in range(len(answer))))
    if kind == "cvrp":
        if not isinstance(answer, list) or not all(isinstance(route, list) and route for route in answer):
            raise ValueError("CVRP answer must be a list of nonempty customer routes")
        coords, demands = instance["coordinates"], instance["demands"]
        customers = [node for route in answer for node in route]
        if any(type(x) is not int for x in customers) or set(customers) != set(coords) - {1} or len(customers) != len(coords) - 1:
            raise ValueError("CVRP routes must visit each customer exactly once")
        total = 0
        for route in answer:
            if sum(demands[node] for node in route) > instance["capacity"]:
                raise ValueError("CVRP capacity exceeded")
            tour = [1, *route, 1]
            total += sum(_distance(coords[a], coords[b]) for a, b in zip(tour, tour[1:]))
        return float(total)
    if kind == "jssp":
        operations = instance["operations"]
        counts = [0] * instance["jobs"]
        by_job = [[] for _ in counts]
        for op in operations:
            by_job[op["job"]].append(op)
        for job_ops in by_job:
            job_ops.sort(key=lambda op: op["step"])
        if not isinstance(answer, list) or len(answer) != len(operations) or any(type(x) is not int or x < 0 or x >= len(counts) for x in answer):
            raise ValueError("JSSP answer must be a job sequence with one entry per operation")
        job_free = [0] * len(counts)
        machine_free = [0] * instance["machines"]
        for job in answer:
            index = counts[job]
            if index >= len(by_job[job]):
                raise ValueError("JSSP job occurs too many times")
            op = by_job[job][index]
            machine = op["machine"]
            end = max(job_free[job], machine_free[machine]) + op["duration"]
            job_free[job] = machine_free[machine] = end
            counts[job] += 1
        if counts != [len(job_ops) for job_ops in by_job]:
            raise ValueError("JSSP answer omits operations")
        return float(max(job_free))
    if kind == "3d-bpp":
        items, size = instance["items"], instance["bin_size"]
        if not isinstance(answer, list) or len(answer) != len(items):
            raise ValueError("3D-BPP answer needs one [bin,x,y,z] placement per item")
        for i, place in enumerate(answer):
            if not isinstance(place, (list, tuple)) or len(place) != 4 or any(type(v) is not int or v < 0 for v in place):
                raise ValueError("3D-BPP placements must be nonnegative integers")
            if any(place[axis + 1] + items[i][axis] > size[axis] for axis in range(3)):
                raise ValueError("3D-BPP item exceeds bin bounds")
            for j in range(i):
                other = answer[j]
                if place[0] == other[0] and all(
                    place[axis + 1] < other[axis + 1] + items[j][axis]
                    and other[axis + 1] < place[axis + 1] + items[i][axis]
                    for axis in range(3)
                ):
                    raise ValueError(f"3D-BPP items {j} and {i} overlap")
        return float(len({place[0] for place in answer}))
    raise ValueError(f"Unknown solver kind: {kind}")


def _run_problem(kind: str, solution_path: str, problem_dir: str,
                 eval_snapshot_path: str, callback_path: str, args: list[str]) -> None:
    solution = Path(solution_path).resolve()
    problem = Path(problem_dir).resolve()
    snapshot = Path(eval_snapshot_path).resolve()
    sys.path.insert(0, str(solution.parent))
    sys.path.insert(1, str(problem))
    sys.path.insert(2, str(problem.parent))
    if kind == "heuristic":
        spec = importlib.util.spec_from_file_location("seed_solution", solution)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot import heuristic solution")
        module = importlib.util.module_from_spec(spec)
        sys.modules["seed_solution"] = module
        spec.loader.exec_module(module)
    sys.argv = [str(problem / "eval.py"), *args]
    namespace = {"__name__": "__main__", "__file__": str(problem / "eval.py")}
    if callback_path != "-":
        namespace["Callbacks"] = runpy.run_path(callback_path)["Callbacks"]
    source = snapshot.read_text(encoding="utf-8")
    exec(compile(source, str(problem / "eval.py"), "exec"), namespace)


def _test_evaluator(kind: str, problem_name: str) -> None:
    """Evaluate a bundled seed through the same path used by research rounds."""
    config = load_config(problem_name=problem_name)
    if config.task.kind != kind:
        raise ValueError(f"{problem_name} is a {config.task.kind} task, not {kind}")
    if not config.task.seed_file:
        raise ValueError(f"{problem_name} has no seed_file")
    seed_path = config.problem_path / config.task.seed_file
    solution = Solution(id=f"{problem_name}_seed", coordinate_agent_id=0, solution_count=0,
                        code=seed_path.read_text(encoding="utf-8"))
    config.output_dir = config.project_root / "runs" / "evaluator_test" / problem_name
    raw_output, metrics, features, score = Evaluator(config).run(solution)
    print(f"======== {kind}: {problem_name} ========")
    print(f"solution: {solution.code_path}")
    print(f"raw output:\n{raw_output}")
    print(f"metrics: {metrics}")
    print(f"features: {features}")
    print(f"score: {score}")
    if solution.evaluation.status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    mode, *arguments = (sys.argv[1:] or ["all"])
    if mode == "problem":
        if len(arguments) < 5:
            raise SystemExit("Internal problem mode requires kind, solution, problem, eval snapshot, and callbacks")
        _run_problem(arguments[0], arguments[1], arguments[2], arguments[3],
                     arguments[4], arguments[5:])
    elif mode == "heuristic":
        if len(arguments) > 1:
            raise SystemExit("Usage: python -m copscientist.evaluator heuristic [problem_name]")
        _test_evaluator("heuristic", arguments[0] if arguments else "tsp_constructive")
    elif mode == "solver":
        if len(arguments) > 1:
            raise SystemExit("Usage: python -m copscientist.evaluator solver [problem_name]")
        _test_evaluator("solver", arguments[0] if arguments else "tsp")
    elif mode == "all" and not arguments:
        _test_evaluator("heuristic", "tsp_constructive")
        _test_evaluator("solver", "tsp")
    else:
        raise SystemExit("Usage: python -m copscientist.evaluator [heuristic [problem_name] | solver [problem_name] | all]")
