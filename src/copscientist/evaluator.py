"""
# Evaluator
Evaluator class is used for evaluating solution functions.

Execute candidate algorithms and return objective values plus diagnostics.
"""

from __future__ import annotations

import json
import ast
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

from .config import Config
from .utils import Evaluation


def _parse_heuristic_result(stdout: str) -> tuple[dict, tuple[int, ...], float]:
    """Read the structured result emitted by a problem's eval.py."""
    lines = [line.strip() for line in stdout.splitlines()]
    if "__SANDBOX_RESULT__" not in lines or "__SANDBOX_SUCCESS__" not in lines:
        raise ValueError("eval.py did not report a successful structured result")

    def field(name: str):
        start, end = f"__{name}_START__", f"__{name}_END__"
        if lines.count(start) != 1 or lines.count(end) != 1:
            raise ValueError(f"eval.py must report exactly one {name} block")
        left, right = lines.index(start), lines.index(end)
        if right != left + 2:
            raise ValueError(f"Malformed {name} block")
        return ast.literal_eval(lines[left + 1])

    metrics = field("METRICS")
    features = field("FEATURES")
    score = field("SCORE")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("eval.py metrics must be a nonempty dict")
    if any(type(key) not in (int, str) or isinstance(value, bool) or
           not isinstance(value, (int, float)) or not math.isfinite(value)
           for key, value in metrics.items()):
        raise ValueError("eval.py metrics must contain finite numeric values")
    if not isinstance(features, tuple) or len(features) != len(metrics) or any(
            type(value) is not int for value in features):
        raise ValueError("eval.py features must be an integer tuple matching metrics")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("eval.py score must be finite")
    return metrics, features, float(score)


class Evaluator:
    """Run existing eval.py scripts or validate solver outputs in subprocesses."""

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def _save_output(code_path: Path, output: str) -> str:
        target = code_path.parent / f"evaluation_{uuid.uuid4().hex[:10]}.txt"
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
        if callbacks is None:
            return "-"
        tree = ast.parse(callbacks)
        if not any(isinstance(node, ast.ClassDef) and node.name == "Callbacks"
                   for node in tree.body):
            raise ValueError("Callback code must define class Callbacks")
        target = code_path.parent / f"callbacks_{uuid.uuid4().hex[:10]}.py"
        target.write_text(callbacks, encoding="utf-8")
        return str(target)

    def evaluate(self, code_path: Path, *, validation: bool = False,
                 callbacks: str | None = None) -> Evaluation:
        try:
            callback_path = self._callback_path(code_path, callbacks)
            if self.config.task.kind == "heuristic":
                result = self._heuristic(code_path, validation, callback_path)
            else:
                result = self._solver(code_path, validation, callback_path)
            result.callbacks = callbacks
            return result
        except (OSError, ValueError, SyntaxError, subprocess.TimeoutExpired) as exc:
            return Evaluation(status="failed", feedback=str(exc)[:3000], callbacks=callbacks)

    def _heuristic(self, code_path: Path, validation: bool, callback_path: str) -> Evaluation:
        args = self.config.task.validation_args if validation else self.config.task.train_args
        if not args:
            raise ValueError("heuristic task requires train_args / validation_args")
        results = []
        outputs = []
        for _ in range(self.config.evaluation_repeats):
            result = self._run(
                [self._python(), "-m", "copscientist.evaluator", "heuristic", str(code_path.parent),
                 str(self.config.problem_path), callback_path, *args],
                self.config.problem_path, self.config.timeout_seconds,
            )
            if result.returncode:
                output = "\n".join(outputs + [result.stdout, result.stderr])
                return Evaluation(status="failed", feedback=(result.stderr or result.stdout)[-3000:],
                                  output_path=self._save_output(code_path, output))
            try:
                results.append(_parse_heuristic_result(result.stdout))
            except (ValueError, SyntaxError) as exc:
                output = "\n".join(outputs + [result.stdout, result.stderr])
                detail = (result.stderr or result.stdout)[-2500:]
                return Evaluation(status="failed", feedback=f"{exc}: {detail}"[-3000:],
                                  output_path=self._save_output(code_path, output))
            outputs.append(result.stdout)
        keys = list(results[0][0])
        if any(list(metrics) != keys for metrics, _, _ in results):
            return Evaluation(status="failed", feedback="Repeated evaluations reported different metric keys",
                              output_path=self._save_output(code_path, "\n".join(outputs)))
        metrics = {key: statistics.mean(result[0][key] for result in results) for key in keys}
        features = tuple(int(value) for value in metrics.values())
        return Evaluation(status="ok", score=statistics.mean(result[2] for result in results),
                          metrics=metrics, features=features,
                          feedback="\n".join(outputs)[-3000:],
                          output_path=self._save_output(code_path, "\n".join(outputs)))

    def _solver(self, code_path: Path, validation: bool, callback_path: str) -> Evaluation:
        paths = []
        patterns = (self.config.task.validation_instances if validation and self.config.task.validation_instances
                    else self.config.task.instances)
        for pattern in patterns:
            paths.extend(self.config.problem_path.glob(pattern))
        paths = sorted(set(path for path in paths if path.is_file()))[:self.config.task.max_instances]
        if not paths:
            raise ValueError("No solver instances match task.instances")
        scores = []
        values = {}
        gaps = {}
        outputs = []
        for path in paths:
            result = self._run(
                [self._python(), "-m", "copscientist.evaluator", "solver", str(code_path),
                 self.config.task.entry_function, str(path), callback_path],
                self.config.problem_path, self.config.timeout_seconds,
            )
            if result.returncode:
                output = "\n".join(outputs + [result.stdout, result.stderr])
                return Evaluation(status="failed", feedback=f"{path.name}: {(result.stderr or result.stdout)[-3000:]}",
                                  output_path=self._save_output(code_path, output))
            outputs.append(f"{path.name}:\n{result.stdout}\n{result.stderr}")
            try:
                payload = json.loads(result.stdout.splitlines()[-1])
                score = float(payload["score"])
            except (IndexError, KeyError, ValueError, json.JSONDecodeError) as exc:
                return Evaluation(status="failed", feedback=f"Invalid result for {path.name}: {exc}",
                                  output_path=self._save_output(code_path, "\n".join(outputs)))
            if not math.isfinite(score):
                return Evaluation(status="failed", feedback=f"Nonfinite score for {path.name}",
                                  output_path=self._save_output(code_path, "\n".join(outputs)))
            name = str(path.relative_to(self.config.problem_path)).replace("\\", "/")
            values[name] = score
            reference = self._reference(path)
            if reference is not None and reference > 0:
                gap = 100 * (score - reference) / reference
                gaps[name] = gap
                scores.append(gap)
            else:
                scores.append(score)
        if gaps and len(gaps) != len(paths):
            return Evaluation(status="failed", feedback="Reference values are missing for some instances; cannot mix gaps and raw objectives")
        return Evaluation(status="ok", score=statistics.mean(scores),
                          metrics={"instances": values, "gaps_percent": gaps,
                                   "score_unit": "mean_gap_percent" if gaps else "mean_objective",
                                   "validation": validation},
                          feedback=(f"Validated {len(paths)} feasible solutions\n" +
                                    "\n".join(outputs))[-3000:],
                          output_path=self._save_output(code_path, "\n".join(outputs)))

    def _reference(self, path: Path) -> float | None:
        kind = self.config.task.entry_function
        if kind == "tsp":
            values_path = self.config.problem_path / "tsplib_euc_2d" / "optimal_values.json"
            records = json.loads(values_path.read_text(encoding="utf-8"))["instances"]
            return records.get(path.name.split(".")[0], {}).get("optimal_length")
        if kind == "cvrp":
            solution = self.config.problem_path / "X-solutions" / f"{path.stem}.sol"
            if solution.is_file():
                match = re.search(r"^Cost\s+(\d+(?:\.\d+)?)", solution.read_text(encoding="utf-8"), re.M)
                return float(match.group(1)) if match else None
        if kind == "jssp":
            records = json.loads((self.config.problem_path / "bks.json").read_text(encoding="utf-8"))
            return next((record["upper_bound"] for record in records
                         if record["instance"] == path.stem), None)
        return None


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


def _run_heuristic(candidate_dir: str, problem_dir: str,
                   callback_path: str, args: list[str]) -> None:
    candidate = Path(candidate_dir).resolve()
    problem = Path(problem_dir).resolve()
    sys.path.insert(0, str(candidate))
    sys.path.insert(1, str(problem))
    # OR-Agent-style evaluators import seed_heuristic, while the current
    # candidate is stored as gpt.py. Bind that import to the candidate file.
    candidate_file = candidate / "gpt.py"
    evaluator_source = (problem / "eval.py").read_text(encoding="utf-8")
    if candidate_file.is_file() and re.search(
            r"^\s*(?:from\s+seed_heuristic\s+import|import\s+seed_heuristic\b)",
            evaluator_source, re.M):
        spec = importlib.util.spec_from_file_location("seed_heuristic", candidate_file)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot import heuristic candidate")
        module = importlib.util.module_from_spec(spec)
        sys.modules["seed_heuristic"] = module
        spec.loader.exec_module(module)
    sys.argv = [str(problem / "eval.py"), *args]
    globals_for_eval = ({"Callbacks": runpy.run_path(callback_path)["Callbacks"]}
                        if callback_path != "-" else {})
    runpy.run_path(str(problem / "eval.py"), run_name="__main__",
                   init_globals=globals_for_eval)


def _run_solver(candidate_path: str, kind: str, instance_path: str,
                callback_path: str) -> None:
    spec = importlib.util.spec_from_file_location("candidate", candidate_path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import candidate")
    module = importlib.util.module_from_spec(spec)
    if callback_path != "-":
        module.Callbacks = runpy.run_path(callback_path)["Callbacks"]
    spec.loader.exec_module(module)
    instance = load_instance(kind, Path(instance_path))
    answer = module.solve(instance)
    score = validate(kind, instance, answer)
    print(json.dumps({"score": score, "name": instance["name"]}))


if __name__ == "__main__":
    mode, *arguments = sys.argv[1:]
    if mode == "heuristic":
        _run_heuristic(arguments[0], arguments[1], arguments[2], arguments[3:])
    elif mode == "solver":
        _run_solver(*arguments)
    else:
        raise SystemExit(f"Unknown evaluator mode: {mode}")
