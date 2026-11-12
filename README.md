# COPScientist

COPScientist is a research workflow for combinatorial optimization. It takes an existing heuristic or solver as a starting point, proposes a new algorithmic idea, implements it, evaluates it, records the outcome, and uses that evidence in the next round. The workflow keeps OR-Agent's research tree and specialized roles while using a persistent ideation and experimentation memory inspired by EvoScientist. It does not use genetic operators, population sampling, or crossover.

## Agents and workflow

1. **COPScientist** loads configuration, owns the shared SQLite database, and coordinates one or more Coordinate Agents.
2. **Coordinate Agent** chooses an unexpanded research-tree node, requests ideas, runs implementation and experiments, and moves to another branch when a node is exhausted.
3. **Ideation Agent** searches scholarly metadata through Crossref and, when configured, the web through Brave Search. It asks the model for a source-grounded literature review, then proposes concrete cross-domain transfers using that review, the tree, and prior memory. Retrieved URLs and search errors are saved.
4. **Coding Agent** implements each idea as a complete Python function and retries after syntax or evaluation failures.
5. **Experiment Agent** runs the implementation in a subprocess, checks feasibility for solver tasks, collects objective values, and summarizes failures and next steps.
6. **Evolution Agent** stores the candidate lineage and two types of memory in SQLite: ideation outcomes and experimentation findings. Each memory entry links back to the candidate, metrics, and cited sources.

The tree expands candidates as refinements of a single parent. `rounds`, `ideas_per_round`, `max_depth`, and `coordinate_agents` bound the work. Selection favors the best unexpanded measured candidate. The best training candidate is evaluated on separate validation instances at the end.

The code follows OR-Agent's role-based module layout: `core.py` starts the run; `coordinate_agent.py`, `ideation_agent.py`, `coding_agent.py`, `experiment_agent.py`, and `evolution_agent.py` each own one role; `flow_graph.py` manages the research tree; `evaluator.py` runs task evaluations and its subprocess entry points; and `research_database.py` stores candidates, sources, and memories. Shared records, model response parsing, and prompt loading live in `utils.py`. Agent system and user prompts are editable text files in `prompts/`, resolved from the COPScientist project root.

## Install and run

Python 3.11 or newer is required. From this directory:

```powershell
python -m pip install -e .
$env:OPENAI_API_KEY = "your-key"
$env:BRAVE_SEARCH_API_KEY = "your-optional-key"
python -m copscientist --problem tsp --rounds 2
```

Configuration follows three layers: the global `config.yaml` sets the default `problem`, model, search, and run budgets; each `problems/heuristics/<name>/settings.yaml` or `problems/solvers/<name>/settings.yaml` defines the problem interface and evaluation instances; command-line options such as `--problem`, `--rounds`, `--model`, and `--timeout-seconds` override them. The global file is loaded from `--config` when specified, otherwise from the current directory, otherwise from the bundled project file. `--init-config` copies the bundled file to the current directory if no config exists there. The resolved configuration is saved as `resolved_config.json` with every run.

The default problem is `tsp_constructive`; install its evaluation dependencies with `python -m pip install -e '.[heuristics]'`. For `mkp_aco`, install `'.[mkp]'`. The chat endpoint is OpenAI-compatible and can be changed through `llm.base_url`, `llm.model`, and `llm.api_key_env`. [Brave Search](https://api-dashboard.search.brave.com/documentation/guides/authentication) is optional unless `search.required = true`; without its key, the skipped web queries are recorded in `literature_review.json`. [Crossref](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) paper metadata is retrieved without an API key. Search results are metadata and snippets, so the literature review must not claim findings absent from those fields.

Available problem settings under `problems/`:

| Type | Names |
| --- | --- |
| Heuristics | `tsp_constructive`, `kp-constructive`, `tsp_aco`, `cvrp_aco`, `mkp_aco`, `bpp_offline_aco`, `bpp_online` |
| Solvers | `tsp`, `cvrp`, `jssp`, `3d-bpp` |

Run output goes to `runs/<problem>/` by default. `result.json` contains the research tree, best training candidate, and validation result. `literature_review.json` contains retrieved references and search errors. `memory.sqlite3` contains the persistent shared memory. Candidate source files remain under `candidates/`. Use `--output-dir` to isolate runs.

## Task contract

Heuristic tasks use the existing `problems/heuristics/<task>/eval.py`. COPScientist copies the original `gpt.py` as the baseline and runs each candidate's own `gpt.py` against the original evaluator and dataset. The evaluator's `Average` output is the training score; the problem settings set `min` or `max`. Existing problem scripts and datasets are not edited during a run.

Solver tasks require `solve(instance)` in a Python file. The supplied baselines in `problems/solvers/*/seed_solver.py` demonstrate the contract:

| Task | `solve` return value | Objective |
| --- | --- | --- |
| TSP | zero-based permutation of cities | TSPLIB EUC_2D tour length |
| CVRP | list of routes containing original customer IDs, depot excluded | EUC_2D route length with capacity checks |
| JSSP | job-ID sequence, each job repeated once per operation | earliest-start makespan respecting job and machine order |
| 3D-BPP | one `[bin, x, y, z]` placement per item | number of bins, fixed orientation and no overlap |

TSP reports a mean gap to the supplied optimal values. CVRP reports a gap to supplied solution costs. JSSP reports a gap to the supplied best-known upper bounds. These are reference gaps, not new optimality proofs. 3D-BPP reports mean bins. A failed or infeasible candidate has no score and cannot be selected as best.

Generated Python runs with the current user's permissions in a timed subprocess. Use a trusted model endpoint and review generated code before running against sensitive local data.

## Tests

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m unittest discover -s tests -v
```
