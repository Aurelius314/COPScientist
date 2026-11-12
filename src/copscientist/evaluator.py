"""Execute candidate algorithms and return objective values plus diagnostics."""

from __future__ import annotations

import json
import gzip
import importlib.util
import math
import os
import re
import runpy
import statistics
import subprocess
import sys
from pathlib import Path

from .config import Config
from .utils import Evaluation


_NUMBER = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _aggregate(stdout: str, mode: str = "mean") -> float:
    lines = [line.strip() for line in stdout.splitlines()]
    values = []
    for index, line in enumerate(lines):
        if "Average" not in line:
            continue
        match = re.search(r"Average[^:]*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", line)
        if match:
            values.append(float(match.group(1)))
        elif index + 1 < len(lines) and _NUMBER.fullmatch(lines[index + 1]):
            values.append(float(lines[index + 1]))
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("eval.py did not report a finite Average")
    return values[-1] if mode == "last" else statistics.mean(values)


class Evaluator:
    """Run existing eval.py scripts or validate solver outputs in subprocesses."""

    def __init__(self, config: Config):
        self.config = config

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        source = str(self.config.project_root / "src")
        env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def _run(self, command: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
        return subprocess.run(command, cwd=cwd, env=self._env(), capture_output=True,
                              text=True, timeout=timeout, check=False)

    def evaluate(self, code_path: Path, *, validation: bool = False) -> Evaluation:
        try:
            if self.config.task.kind == "heuristic":
                return self._heuristic(code_path, validation)
            return self._solver(code_path, validation)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return Evaluation(status="failed", feedback=str(exc)[:3000])

    def _heuristic(self, code_path: Path, validation: bool) -> Evaluation:
        args = self.config.task.validation_args if validation else self.config.task.train_args
        if not args:
            raise ValueError("heuristic task requires train_args / validation_args")
        scores = []
        outputs = []
        for _ in range(self.config.experiment_repeats):
            result = self._run(
                [sys.executable, "-m", "copscientist.evaluator", "heuristic", str(code_path.parent),
                 str(self.config.problem_path), *args],
                self.config.problem_path, self.config.timeout_seconds,
            )
            if result.returncode:
                return Evaluation(status="failed", feedback=(result.stderr or result.stdout)[-3000:])
            scores.append(_aggregate(result.stdout, self.config.task.aggregate))
            outputs.append(result.stdout[-1500:])
        return Evaluation(status="ok", score=statistics.mean(scores),
                          metrics={"scores": scores, "validation": validation},
                          feedback="\n".join(outputs)[-3000:])

    def _solver(self, code_path: Path, validation: bool) -> Evaluation:
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
        for path in paths:
            result = self._run(
                [sys.executable, "-m", "copscientist.evaluator", "solver", str(code_path),
                 self.config.task.entry_function, str(path)],
                self.config.problem_path, self.config.timeout_seconds,
            )
            if result.returncode:
                return Evaluation(status="failed", feedback=f"{path.name}: {(result.stderr or result.stdout)[-3000:]}")
            try:
                payload = json.loads(result.stdout.splitlines()[-1])
                score = float(payload["score"])
            except (IndexError, KeyError, ValueError, json.JSONDecodeError) as exc:
                return Evaluation(status="failed", feedback=f"Invalid result for {path.name}: {exc}")
            if not math.isfinite(score):
                return Evaluation(status="failed", feedback=f"Nonfinite score for {path.name}")
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
                          feedback=f"Validated {len(paths)} feasible solutions")

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


def _run_heuristic(candidate_dir: str, problem_dir: str, args: list[str]) -> None:
    candidate = Path(candidate_dir).resolve()
    problem = Path(problem_dir).resolve()
    sys.path.insert(0, str(candidate))
    sys.path.insert(1, str(problem))
    sys.argv = [str(problem / "eval.py"), *args]
    runpy.run_path(str(problem / "eval.py"), run_name="__main__")


def _run_solver(candidate_path: str, kind: str, instance_path: str) -> None:
    spec = importlib.util.spec_from_file_location("candidate", candidate_path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import candidate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = load_instance(kind, Path(instance_path))
    answer = module.solve(instance)
    score = validate(kind, instance, answer)
    print(json.dumps({"score": score, "name": instance["name"]}))


if __name__ == "__main__":
    mode, *arguments = sys.argv[1:]
    if mode == "heuristic":
        _run_heuristic(arguments[0], arguments[1], arguments[2:])
    elif mode == "solver":
        _run_solver(*arguments)
    else:
        raise SystemExit(f"Unknown evaluator mode: {mode}")
