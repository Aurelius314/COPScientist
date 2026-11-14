"""Experiment decisions and evidence summaries around the evaluator."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copscientist.config import Config, load_config
from copscientist.evaluator import Evaluator
from copscientist.utils import (Evaluation, Idea, LLMClient, Model, Solution, better,
                                extract_diff_block, extract_json_block,
                                extract_python_code, extract_thinking, load_prompt,
                                parse_json, update_code)


class ExperimentAgent:
    def __init__(self, config: Config, model: Model | None = None,
                 evaluator: Evaluator | None = None,
                 summary_model: Model | None = None):
        self.config = config
        self.model = model
        self.summary_model = summary_model or model
        self.evaluator = evaluator or Evaluator(config)
        self.system_experiment_reflection_prompt = load_prompt("system_experiment_reflection_copscientist.txt")
        self.user_experiment_reflection_prompt = load_prompt("user_experiment_reflection_copscientist.txt")
        self.system_experiment_summary_prompt = load_prompt("system_experiment_summary_copscientist.txt")
        self.user_experiment_summary_prompt = load_prompt("user_experiment_summary_copscientist.txt")
        knowledge = config.problem_path / "external_knowledge.txt"
        self.external_knowledge = (knowledge.read_text(encoding="utf-8")
                                   if knowledge.is_file() else "")
        callbacks_description = config.problem_path / "callbacks_description.txt"
        self.callbacks_description = (callbacks_description.read_text(encoding="utf-8")
                                      if callbacks_description.is_file() else "")
        if config.verbose:
            print(">>>[ExperimentAgent] Initialized")

    def evaluate(self, code_path: Path, *, validation: bool = False,
                 callbacks: str | None = None) -> Evaluation:
        return self.evaluator.evaluate(code_path, validation=validation,
                                       callbacks=callbacks)

    def evaluate_solution(self, solution: Solution, *, validation: bool = False,
                          callbacks: str | None = None) -> Solution:
        self.evaluator.run(solution, callbacks=callbacks, validation=validation)
        return solution

    def reflect(self, idea: Idea, code: str, evaluation: Evaluation,
                parent_score: float | None, history: list[dict],
                long_term_reflection: str = "",
                parent: Solution | None = None,
                compact_summaries: list[str] | None = None
                ) -> tuple[str, str | None, str | None, str | None, dict | None]:
        """Extract analysis, code, callbacks, and termination from one response."""
        if self.model is None:
            return "", None, None, None, {"termination": "yes"}
        latest_output = evaluation.feedback
        if evaluation.output_path and Path(evaluation.output_path).is_file():
            latest_output = Path(evaluation.output_path).read_text(encoding="utf-8")[-12000:]
        context = {"idea": asdict(idea), "code": code[-12000:],
                   "evaluation": asdict(evaluation), "parent_score": parent_score,
                   "parent_solution": (parent.idea_performance_summary_str() if parent else None),
                   "latest_experiment_result": latest_output,
                   "objective": self.config.task.objective,
                   "long_term_reflection": long_term_reflection,
                   "external_knowledge": (self.external_knowledge if not self.config.external_knowledge_persistence_disabled
                                          else ""),
                   "callbacks_description": self.callbacks_description,
                   "current_callbacks": evaluation.callbacks,
                   "progressive_summary": "\n\n".join(compact_summaries or []),
                   "history": [{"step": step["step"], "score": step["evaluation"]["score"],
                                "status": step["evaluation"]["status"],
                                "action": step["action"]}
                               for step in history]}
        for name in ("problem_description", "function_description", "evaluation_description"):
            source = self.config.problem_path / f"{name}.txt"
            if source.is_file() and (name != "evaluation_description"
                                     or not self.config.evaluation_description_disabled):
                context[name] = source.read_text(encoding="utf-8")[:8000]
        response = self.model.complete(
            self.system_experiment_reflection_prompt,
            self.user_experiment_reflection_prompt.replace(
                "{{context}}", json.dumps(context, ensure_ascii=False)),
        )
        return (response, extract_thinking(response), extract_diff_block(response),
                extract_python_code(response), extract_json_block(response))

    @staticmethod
    def generate_progressive_summary(history: list[dict], objective: str) -> str:
        """Keep OR-Agent's score trend, metrics, and reflected insights before trimming history."""
        first_score = history[0]["evaluation"]["score"]
        last_score = history[-1]["evaluation"]["score"]
        scores = [step["evaluation"]["score"] for step in history
                  if step["evaluation"]["score"] is not None]
        best_score = (min(scores) if objective == "min" else max(scores)) if scores else None
        lines = [f"Progressive Summary for experiments #{history[0]['step'] + 1} "
                 f"~ #{history[-1]['step'] + 1}:",
                 f"- Initial score: {first_score}", f"- Current score: {last_score}",
                 f"- Best score so far: {best_score}"]
        if first_score is not None and last_score is not None:
            improvement = (last_score - first_score if objective == "max"
                           else first_score - last_score)
            relative = improvement / abs(first_score) if first_score else improvement
            lines.extend([f"- Absolute improvement: {improvement:.4f}",
                          f"- Relative improvement: {relative:.2%}"])
        lines.append("Key insights from recent experiments:")
        for step in history:
            result = step["evaluation"]
            insight = extract_thinking(step["action"]) if step["action"] else None
            lines.append(f"- Experiment #{step['step'] + 1}: score={result['score']}; "
                         f"metrics={result.get('metrics', {})}; insight={insight or '(none)'}")
        return "\n".join(lines)

    def run(self, idea: Idea, code_path: Path, initial: Evaluation,
            parent_score: float | None, long_term_reflection: str = "",
            max_repeats: int | None = None,
            parent: Solution | None = None) -> tuple[Evaluation, list[dict], str, list[str]]:
        """Evaluate, reflect, revise, and restore the best tested code version."""
        if self.config.verbose:
            print(f"\n>>>[ExperimentAgent] Starts to work on {code_path.name}...")

        # =====0. Experiment preparation=====
        original_code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
        best = initial if max_repeats == 0 else None
        best_code = original_code
        current_callbacks = initial.callbacks
        history: list[dict] = []
        compact_summaries: list[str] = []
        repeats = max_repeats if max_repeats is not None else self.config.experiment_repeats
        if repeats < 0:
            raise ValueError("Experiment budget cannot be negative")
        if repeats == 0:
            best = initial
        if self.config.verbose:
            print(f">>>[ExperimentAgent] Experiment budget: {repeats}")

        # =====1. Experiment loop=====
        pending_revision = False
        for experiment_count in range(1, repeats + 1):
            # -----1.1. Evaluate the current code, including on the first round.-----
            code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
            if not code:
                break
            current = self.evaluator.evaluate(code_path, callbacks=current_callbacks)
            pending_revision = False
            snapshot = code_path.with_name(f"experiment_{experiment_count - 1}.py")
            snapshot.write_text(code, encoding="utf-8")
            record = {"step": experiment_count - 1, "code_path": str(snapshot),
                      "evaluation": asdict(current), "action": None}
            history.append(record)
            if best is None or (current.status == "ok" and
                                better(current.score, best.score, self.config.task.objective)):
                best, best_code = current, code
            if self.config.verbose:
                print(f">>>[ExperimentAgent] Experiment {experiment_count}/{repeats}: "
                      f"{current.status}, score={current.score}")

            # -----1.2. Reflect; one response may contain code, callbacks, and termination.-----
            try:
                response, thinking, code_diff, callbacks, termination = self.reflect(
                    idea, code, current, parent_score, history[:-1],
                    long_term_reflection, parent=parent,
                    compact_summaries=compact_summaries)
            except (ValueError, OSError) as exc:
                response, thinking, code_diff, callbacks = str(exc), None, None, None
                termination = {"termination": "yes"}
            record["action"] = response
            if self.config.verbose:
                print(f">>>[ExperimentAgent] Analysis: {thinking or '(none)'}")

            # -----1.3. Compress old detailed history at OR-Agent's configured cadence.-----
            if (self.config.progressive_summarization and experiment_count > 1
                    and experiment_count % self.config.summarization_interval == 0):
                compact = self.generate_progressive_summary(history, self.config.task.objective)
                compact_summaries.append(compact)
                keep = self.config.compact_history_length
                history = history[-keep:] if keep > 0 else []
                if self.config.verbose:
                    print(f">>>[ExperimentAgent] Compressed history after experiment {experiment_count}")

            # -----1.4. Apply code and callbacks independently, then honor termination.-----
            details = self.config.run_path / "details"
            if code_diff:
                try:
                    revised = update_code(code, code_diff)
                    code_path.write_text(revised, encoding="utf-8")
                    details.mkdir(parents=True, exist_ok=True)
                    (details / f"{code_path.parent.name}_codediff_{experiment_count}.txt").write_text(
                        code_diff, encoding="utf-8")
                    pending_revision = True
                    if self.config.verbose:
                        print(f">>>[ExperimentAgent] Updated code in experiment {experiment_count}")
                except (OSError, ValueError, SyntaxError) as exc:
                    record["revision_error"] = str(exc)
                    if self.config.verbose:
                        print(f">>>[ExperimentAgent] Code update failed: {exc}")
            if callbacks:
                current_callbacks = callbacks
                details.mkdir(parents=True, exist_ok=True)
                (details / f"{code_path.parent.name}_callbacks_{experiment_count}.txt").write_text(
                    callbacks, encoding="utf-8")
                pending_revision = True
                if self.config.verbose:
                    print(f">>>[ExperimentAgent] Updated callbacks in experiment {experiment_count}")
            decision = str((termination or {}).get("termination", "no")).lower().strip()
            if self.config.verbose:
                print(f">>>[ExperimentAgent] Termination: {decision}")
            if decision == "yes":
                break

        # =====2. Evaluate a final untested revision and restore the best version.=====
        if pending_revision:
            final_code = code_path.read_text(encoding="utf-8")
            current = self.evaluator.evaluate(code_path, callbacks=current_callbacks)
            final_snapshot = code_path.with_name(f"experiment_{experiment_count}.py")
            final_snapshot.write_text(final_code, encoding="utf-8")
            history.append({"step": experiment_count, "code_path": str(final_snapshot),
                            "evaluation": asdict(current), "action": None})
            if best is None or (current.status == "ok" and
                                better(current.score, best.score, self.config.task.objective)):
                best, best_code = current, final_code
        if best is None:
            best = initial
        if best_code and code_path.exists():
            if self.config.verbose and code_path.read_text(encoding="utf-8") != best_code:
                print(f">>>[ExperimentAgent] Restoring best tested code: score={best.score}")
            code_path.write_text(best_code, encoding="utf-8")

        # -----Summarize all attempts, including failed revisions.-----
        summary = self.summarize(idea, best, parent_score, history,
                                 long_term_reflection, best_code, parent,
                                 compact_summaries, original_code)
        if self.config.verbose:
            print(f">>>[ExperimentAgent] Finished {code_path.name}: "
                  f"{best.status}, score={best.score}")
        return best, history, summary, compact_summaries

    def summarize(self, idea: Idea, evaluation: Evaluation,
                  parent_score: float | None, history: list[dict] | None = None,
                  long_term_reflection: str = "", best_code: str = "",
                  parent: Solution | None = None,
                  compact_summaries: list[str] | None = None,
                  original_code: str = "") -> str:
        if self.summary_model is None:
            return evaluation.feedback[:1200]
        try:
            records = history or []
            if not original_code and records:
                original_code = Path(records[0]["code_path"]).read_text(encoding="utf-8")
            context = {"idea": asdict(idea), "evaluation": asdict(evaluation),
                       "parent_score": parent_score, "objective": self.config.task.objective,
                       "parent_solution": (parent.idea_performance_summary_str() if parent else None),
                       "long_term_reflection": long_term_reflection,
                       "original_code": original_code[:8000], "best_code": best_code[:8000],
                       "compact_summaries": compact_summaries or [],
                       "history": [{"step": step["step"], "evaluation": step["evaluation"],
                                    "action": step["action"]}
                                   for step in records]}
            system = self.system_experiment_summary_prompt
            user = self.user_experiment_summary_prompt.replace(
                "{{context}}", json.dumps(context, ensure_ascii=False))
            for _ in range(5):
                try:
                    data = parse_json(self.summary_model.complete(system, user))
                except (ValueError, TypeError):
                    continue
                if isinstance(data.get("summary"), str) and data["summary"].strip():
                    return data["summary"].strip()
                if any(data.get(key) for key in ("diagnosis", "evidence", "next_step")):
                    return json.dumps({key: str(data.get(key, "")) for key in
                                       ("diagnosis", "evidence", "next_step")}, ensure_ascii=False)
            raise ValueError("No valid experiment summary after five responses")
        except Exception as exc:
            return f"Experiment summary unavailable: {exc}; raw feedback: {evaluation.feedback[:1000]}"


if __name__ == "__main__":
    # For debugging purposes: evaluate a seed, revise it, and inspect the final summary.
    import argparse
    import uuid

    parser = argparse.ArgumentParser(description="Run one Experiment Agent cycle on a seed solution.")
    parser.add_argument("--problem", default="tsp_constructive")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-repeats", type=int)
    parser.add_argument("--offline", action="store_true",
                        help="Exercise a failed revision and best-version restoration without an LLM")
    args = parser.parse_args()

    config = load_config(args.config, args.problem)
    config.output_dir = (args.output_dir or config.project_root / "runs" / "experiment_agent_test"
                         / config.task.name)
    if not config.task.seed_file:
        parser.error(f"{config.task.name} has no seed_file")
    seed_code = (config.problem_path / config.task.seed_file).read_text(encoding="utf-8")

    if args.offline:
        entry = "solve" if config.task.kind == "solver" else config.task.entry_function
        broken_code = (f"def {entry}(*args, **kwargs):\n"
                       "    raise RuntimeError('offline experiment test')\n")

        class OfflineModel:
            def __init__(self):
                self.reflections = 0

            def complete(self, system: str, user: str) -> str:
                if "final report" in system.lower():
                    return json.dumps({"summary": "The failed revision was measured and the seed restored."})
                self.reflections += 1
                if self.reflections == 1:
                    diff = ("<<<<<<< SEARCH\n" + seed_code.rstrip() + "\n=======\n"
                            + broken_code.rstrip() + "\n>>>>>>> REPLACE")
                    return ("<thinking>Test failure recovery.</thinking>\n```diff\n" + diff
                            + "\n```\n```json\n{\"termination\": \"no\"}\n```")
                return ("<thinking>The revision failed; restore the seed.</thinking>\n"
                        "```json\n{\"termination\": \"yes\"}\n```")

        reflect_model: Model = OfflineModel()
        summary_model: Model = reflect_model
        repeats = args.max_repeats if args.max_repeats is not None else 2
    else:
        def configured_model(role: str) -> LLMClient:
            settings = config.role_models.get(role, {})
            return LLMClient(
                config=config,
                llm_provider=str(settings.get("llm_provider", config.llm_provider)),
                model_name=str(settings.get("model", config.model)),
                base_url=settings.get("base_url", config.base_url),
                api_key_env=settings.get("api_key_env", config.api_key_env),
                temperature=float(settings.get("temperature", 0.4)),
                max_retries=int(settings.get("max_retries", config.model_max_retries)),
            )

        reflect_model = configured_model("experiment_reflect")
        summary_model = configured_model("experiment_summarize")
        repeats = args.max_repeats

    evaluator = Evaluator(config)
    seed_solution = Solution(
        id=f"seed_{uuid.uuid4().hex[:8]}", round=0,
        coordinate_agent_id=0, solution_count=0,
        idea=Idea("Seed experiment", "Measure the supplied baseline",
                  "Start from the seed implementation", "Find a better score"),
        code=seed_code,
    )
    evaluator.run(seed_solution)
    print(f">>>[ExperimentAgent test] Initial: {seed_solution.evaluation.status}, "
          f"score={seed_solution.score}")
    if seed_solution.evaluation.status != "ok":
        raise SystemExit(f"Seed evaluation failed: {seed_solution.evaluation.feedback}")

    agent = ExperimentAgent(config, reflect_model, evaluator, summary_model)
    best, history, summary, compact_summaries = agent.run(
        seed_solution.idea, Path(seed_solution.code_path), seed_solution.evaluation,
        parent_score=None, max_repeats=repeats,
    )
    print(f">>>[ExperimentAgent test] Final: {best.status}, score={best.score}; "
          f"recorded experiments={len(history)}")
    print(f">>>[ExperimentAgent test] Summary: {summary}")
    print(f">>>[ExperimentAgent test] Compact summaries: {len(compact_summaries)}")
    print(f">>>[ExperimentAgent test] Code: {seed_solution.code_path}")
    if best.status != "ok":
        raise SystemExit(1)
    if args.offline and Path(seed_solution.code_path).read_text(encoding="utf-8") != seed_code:
        raise SystemExit("Offline test failed to restore the seed code")
