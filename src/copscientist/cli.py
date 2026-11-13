"""Command line interface for COPScientist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config, DEFAULT_CONFIG, PROJECT_ROOT, load_config
from .core import COPScientist


def main() -> int:
    parser = argparse.ArgumentParser(prog="copscientist")
    parser.add_argument("--init-config", action="store_true", help="Copy the default global config to the current directory")
    parser.add_argument("--config", type=Path, help="Global YAML; defaults to ./config.yaml, then the bundled config")
    parser.add_argument("--checkpoint", help="Resume from checkpoints/<name>")
    parser.add_argument("--problem", help="Problem directory name under problems/heuristics or problems/solvers")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--coordinate-agents", type=int)
    parser.add_argument("--ideas-per-round", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--debug-attempts", type=int)
    parser.add_argument("--experiment-repeats", type=int)
    parser.add_argument("--evaluation-repeats", type=int)
    parser.add_argument("--max-evolutions", type=int)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--autosave-interval-minutes", type=float)
    parser.add_argument("--dynamic-experiment-allocation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--min-experiment-repeats", type=int)
    parser.add_argument("--max-experiment-repeats-cap", type=int)
    parser.add_argument("--improvement-threshold", type=float)
    parser.add_argument("--elitist-experiment-factor", type=float)
    parser.add_argument("--progressive-summarization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--summarization-interval", type=int)
    parser.add_argument("--compact-history-length", type=int)
    parser.add_argument("--reflection-period", type=int)
    parser.add_argument("--reflection-clearance-period", type=int)
    parser.add_argument("--reflection-compression", type=int)
    parser.add_argument("--external-knowledge-persistence-disabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--evaluation-description-disabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ideas-coordinated-generation-disabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--python-path")
    parser.add_argument("--model")
    parser.add_argument("--llm-provider")
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
                 if key not in {"init_config", "config", "problem", "checkpoint"}
                 and value is not None}
    if args.checkpoint:
        if Path(args.checkpoint).name != args.checkpoint or args.checkpoint in {".", ".."}:
            parser.error("Checkpoint must be a single directory name")
        incompatible = [key for key in overrides if key not in {"rounds", "max_evolutions"}]
        if args.problem or args.config or incompatible:
            parser.error("Only --rounds and --max-evolutions may override checkpoint settings")
        checkpoint_path = PROJECT_ROOT / "checkpoints" / args.checkpoint / "config.json"
        config = Config.from_dict(json.loads(checkpoint_path.read_text(encoding="utf-8")))
        if args.rounds is not None:
            config.rounds = args.rounds
        if args.max_evolutions is not None:
            config.max_evolutions = args.max_evolutions
    else:
        config = load_config(args.config, args.problem, overrides)
    best = COPScientist(config, checkpoint=args.checkpoint).run()
    print(json.dumps({"problem": config.task.name, "best_id": best.id if best else None,
                      "score": best.evaluation.score if best else None,
                      "run_path": str(config.run_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
