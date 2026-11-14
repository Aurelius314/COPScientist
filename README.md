# COPScientist

COPScientist is a research workflow for combinatorial optimization. It takes an existing heuristic or solver as a starting point, proposes a new algorithmic idea, implements it, evaluates it, records the outcome, and uses that evidence in the next round. The base workflow follows OR-Agent's research tree and specialized roles. It does not use genetic operators, population sampling, or crossover. Optional literature search and EvoScientist-style memory are disabled in the base configuration.

## Agents and workflow

1. **COPScientist** loads configuration, owns the shared SQLite database, and coordinates one or more Coordinate Agents.
2. **Coordinate Agent** starts each research round from a high-scoring tested solution, retries the root stage up to five times if no feasible child is found, expands the best unfinished leaf, retains feasible root branches even after a temporary downgrade, and requires improvement deeper in the tree. It maintains a long-term reflection across experiments.
3. **Ideation Agent** uses the parent, research tree, and long-term reflection to propose refinements. With `search.enabled: true`, it also runs a bounded web and literature survey before the existing idea-generation loop.
4. **Coding Agent** generates code, evaluates it, and performs bounded debugging using SEARCH/REPLACE diffs or a complete corrected module.
5. **Experiment Agent** evaluates, reflects, makes bounded local code or callback revisions, records intermediate versions, raw evaluator output and outcomes, restores the best tested version, and summarizes the experiment. It supports OR-Agent's score-dependent experiment allocation and progressive history summaries.
6. **Evolution Agent** stores the solution lineage in SQLite. When `evolution.enabled: true`, it also distills ideation outcomes and experimentation findings into reusable memory with links to solution metrics and sources.

`rounds`, `ideas_per_round`, `max_depth`, and `coordinate_agents` bound tree exploration. `max_evolutions` tracks model responses and solution evaluations and is checked between rounds, as in OR-Agent. `experiment_repeats` sets the base experiment budget; dynamic allocation uses the parent and child scores, a minimum, a cap, an improvement threshold, and an elitist factor. `evaluation_repeats` controls repeated measurements of the same heuristic version. Each round selects a root from the highest-scoring tested solutions without population, island, or crossover logic. Only accepted leaves join the cross-round root pool. The best training solution is evaluated on separate validation instances at the end.

The code follows OR-Agent's role-based module layout: `core.py` starts the run; `coordinate_agent.py`, `ideation_agent.py`, `coding_agent.py`, `experiment_agent.py`, and `evolution_agent.py` each own one role; `flow_graph.py` manages the research tree; `evaluator.py` runs task evaluations and its subprocess entry points; and `research_database.py` stores solutions, sources, and memories. `utils.py` defines the shared `Solution` record with code, output, metrics, features, score, summary, and intermediate experiments, plus OR-Agent-style identity and history methods. The `Evaluation` and file-path fields preserve saved-run metadata. Shared model response parsing and prompt loading also live in `utils.py`. Agent system and user prompts are editable text files in `prompts/`, resolved from the COPScientist project root.

## Install and run

Python 3.11 or newer is required. From this directory:

```powershell
python -m pip install -e .
$env:OPENAI_API_KEY = "your-key"
$env:TAVILY_API_KEY = "your-optional-key"
python -m copscientist --problem tsp --rounds 2
```

`autosave_interval_minutes` saves a complete checkpoint after a round when the interval has elapsed; set it to `0` to save after every round. Checkpoints live in `checkpoints/<name>/` and include resolved configuration, solution source and evaluator output, SQLite memory, agent reflections, usage counters, and completed-round progress. Resume with `python -m copscientist --checkpoint <name>`. To continue beyond the saved target, add `--rounds <new-total>`; `--max-evolutions` may also increase the budget. Python callers can use `scientist.save("name")` after a run. Checkpoint configuration and problem identity are preserved on resume.

The `llm.llm_provider` setting selects `openai`, `qwen`, `deepseek`, or a custom OpenAI-compatible endpoint; custom providers require `base_url` and `api_key_env`. `LLMClient` exposes OR-Agent-style `chat(messages)` and `multi_chat(messages_list, n)` methods and adapts them to the agents' `complete(system, user)` calls. The `llm.roles` mapping optionally selects a different `llm_provider`, `model`, `base_url`, `api_key_env`, `temperature`, and `max_retries` for `coordinate`, `ideation`, `coding`, `experiment_reflect`, `experiment_summarize`, and `literature_review`. `run.python_path` and `run.eval_env_vars` customize the evaluator subprocess. `ideas_coordinated_generation_disabled`, `evaluation_description_disabled`, `progressive_summarization`, `reflection_period`, `reflection_clearance_period`, and `reflection_compression` expose the corresponding OR-Agent controls.

Configuration follows three layers: the global `config.yaml` sets the default `problem`, model, search, and run budgets; each `problems/heuristics/<name>/settings.yaml` or `problems/solvers/<name>/settings.yaml` defines the problem interface and evaluation instances; command-line options such as `--problem`, `--rounds`, `--model`, and `--timeout-seconds` override them. The global file is loaded from `--config` when specified, otherwise from the current directory, otherwise from the bundled project file. `--init-config` copies the bundled file to the current directory if no config exists there. The resolved configuration is saved as `resolved_config.json` with every run.

The default problem is `tsp_constructive`; install its evaluation dependencies with `python -m pip install -e '.[heuristics]'`. For `mkp_aco`, install `'.[mkp]'`. The chat endpoint is OpenAI-compatible and can be changed through `llm.base_url`, `llm.model`, and `llm.api_key_env`. Search is disabled in the base configuration. When enabled, web search uses `TAVILY_API_KEY` as in EvoScientist; [Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) supplies paper metadata without a search key. The survey fetches up to `search.max_pages` source pages, makes up to `search.max_queries` broad and focused queries, and records unavailable pages and skipped searches in `literature_review.json`. Set `search.required: true` to stop when usable source text cannot be obtained.

Available problem settings under `problems/`:

| Type | Names |
| --- | --- |
| Heuristics | `tsp_constructive`, `kp_constructive`, `tsp_aco`, `cvrp_aco`, `mkp_aco`, `bpp_offline_aco`, `bpp_online` |
| Solvers | `tsp`, `cvrp`, `jssp`, `3d-bpp` |

Run output goes to `runs/<problem>/` by default. `result.json` contains all solutions, the best training solution, validation result, root pool, and usage counts. `results.json` records best-score improvements and `results_detailed.json` records every completed round. `details/coordinate_*_tree.json` records each round's tree; code revisions are also saved in `details/`. `literature_review.json` contains retrieved references and search errors. `memory.sqlite3` contains the persistent shared memory. Solution artifacts live under `solutions/<heuristic|solver>/<solution-id>/`: `<solution-id>_solution.py` holds the algorithm, `<solution-id>_eval.py` holds the evaluator snapshot, and `<solution-id>_stdout_<run-id>.txt` holds each evaluation's output. Use `--output-dir` to isolate runs.

## Task contract

Heuristic tasks use `problems/heuristics/<task>/eval.py`. As in OR-Agent, COPScientist reads the seed script into `Solution.code`; Evaluator then writes a separate ID-named algorithm file instead of copying the seed file directly. Generated solutions follow the same naming rule. Each evaluator calls the solution function on problem-specific train or validation instances and emits the shared `__METRICS_*__`, `__FEATURES_*__`, and `__SCORE_*__` blocks. Bundled evaluators usually discretize metrics into features and average metrics for the score; the protocol also accepts nested metrics, independent scores, and optional features. Repeated runs average compatible numeric metrics and scores. `Evaluator.run(solution, callbacks=...)` evaluates a `Solution`, saves raw output and the evaluation script snapshot, updates the solution record, and returns `(raw_output, metrics, features, score)`. The path-based `evaluate(path)` remains available to Coding and Experiment Agents. Evaluation executes the saved script snapshot with the original problem directory as its data location. Existing problem scripts and datasets are not edited during a run.

Callback exploration is available when a problem provides `callbacks_description.txt` and its evaluator invokes a `Callbacks` class through the documented event API. Experiment Agent can generate or revise that class; COPScientist loads it in the evaluation subprocess and records its output. The bundled problems do not currently define callback events, so their experiments revise solution code. The callback path is covered by a temporary-problem integration test.

Solver tasks require `solve(instance)` in a Python file. The supplied baselines in `problems/solvers/*/seed_solution.py` demonstrate the contract:

| Task | `solve` return value | Objective |
| --- | --- | --- |
| TSP | zero-based permutation of cities | TSPLIB EUC_2D tour length |
| CVRP | list of routes containing original customer IDs, depot excluded | EUC_2D route length with capacity checks |
| JSSP | job-ID sequence, each job repeated once per operation | earliest-start makespan respecting job and machine order |
| 3D-BPP | one `[bin, x, y, z]` placement per item | number of bins, fixed orientation and no overlap |

TSP reports a mean gap to the supplied optimal values. CVRP reports a gap to supplied solution costs. JSSP reports a gap to the supplied best-known upper bounds. These are reference gaps, not new optimality proofs. 3D-BPP reports mean bins. A failed or infeasible solution has no score and cannot be selected as best.

Generated Python runs with the current user's permissions in a timed subprocess. Use a trusted model endpoint and review generated code before running against sensitive local data.

## Tests

Run the evaluator's bundled seed checks separately for a heuristic or solver task. Omitting the mode runs both default checks. Pass another problem name after the mode to test its seed.

```powershell
python src/copscientist/evaluator.py heuristic
python src/copscientist/evaluator.py solver
python src/copscientist/evaluator.py heuristic bpp_online
```

Outputs are saved under `runs/evaluator_test/<problem>/`.

To exercise the Coding Agent's full seed → generate → evaluate → debug path without an API key, use its deterministic offline test. Omit `--offline` to use the configured model, and pass `--idea` to change the child idea.

```powershell
python src/copscientist/coding_agent.py --offline
python src/copscientist/coding_agent.py --offline --problem tsp
```

Coding Agent test artifacts are saved under `runs/coding_agent_test/<problem>/`.

The Experiment Agent has a matching entry point. Its offline test makes a failing local revision and confirms that the best tested seed is restored. Omit `--offline` to use the configured reflection and summary models.

```powershell
python src/copscientist/experiment_agent.py --offline
python src/copscientist/experiment_agent.py --offline --problem tsp
```

Experiment artifacts are saved under `runs/experiment_agent_test/<problem>/`.

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m unittest discover -s tests -v
```
