# JSSP benchmark instances

This directory contains two classic job shop scheduling benchmark families, copied from the [ScheduleOpt/benchmarks jobshop JSON instances](https://github.com/ScheduleOpt/benchmarks/tree/main/jobshop/instances/json):

- `Taillard/`: 80 Taillard (1993) instances, `ta01js.json` through `ta80js.json`.
- `DMU/`: 80 Demirkol, Mehta, and Uzsoy (1998) instances, `dmu01.json` through `dmu80.json`.

Each JSON file describes its jobs, machines, and operations, including each operation's machine and processing duration. The two families contain 160 instances in total.

`bks.json` contains the 160 matching best-known-solution records, selected from
[ScheduleOpt's BKS archive](https://github.com/ScheduleOpt/benchmarks/blob/main/jobshop/solutions/bks.json)
on 2026-11-12. Match each record's `instance` to an instance filename without
`.json`. Use `upper_bound` as the best-known makespan when calculating a gap;
`status` and `lower_bound` describe the reported closure state; `closed` does
not necessarily mean an independently reproduced proof. These bounds can
change as better solutions or proofs are published.

`seed_solver.py` uses the Most Work Remaining (MWKR) dispatching rule: at each
step it selects the job with the largest sum of processing times in its
unscheduled operations (ties go to the smaller job ID). This follows the greedy
dispatching baseline in [Runarsson, Schoenauer, and Sebag, *Pilot, Rollout and
Monte Carlo Tree Search Methods for Job Shop Scheduling*, Algorithm 1](https://www.cmap.polytechnique.fr/~nikolaus.hansen/proceedings/2012/LION/PDF/p87.pdf).
MWKR is also a reported baseline in the [supplement to Zhang et al.,
*Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning*
(NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Supplemental.pdf).
