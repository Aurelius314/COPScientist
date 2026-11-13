"""Command line interface for COPScientist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config
from .core import COPScientist


def main() -> int:
    parser = argparse.ArgumentParser(prog="copscientist")
    parser.add_argument("--init-config", action="store_true", help="Copy the default global config to the current directory")
    parser.add_argument("--config", type=Path, help="Global YAML; defaults to ./config.yaml, then the bundled config")
    parser.add_argument("--problem", help="Problem directory name under problems/heuristics or problems/solvers")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--coordinate-agents", type=int)
    parser.add_argument("--ideas-per-round", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--debug-attempts", type=int)
    parser.add_argument("--experiment-repeats", type=int)
    parser.add_argument("--evaluation-repeats", type=int)
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--search-required", action="store_true", default=None)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.init_config:
        destination = Path.cwd() / "config.yaml"
        if destination.exists():
            parser.error(f"Configuration already exists: {destination}")
        destination.write_text(DEFAULT_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
        print(destination)
        return 0
    overrides = {key: value for key, value in vars(args).items()
                 if key not in {"init_config", "config", "problem"} and value is not None}
    config = load_config(args.config, args.problem, overrides)
    best = COPScientist(config).run()
    print(json.dumps({"problem": config.task.name, "best_id": best.id if best else None,
                      "score": best.evaluation.score if best else None,
                      "run_path": str(config.run_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
