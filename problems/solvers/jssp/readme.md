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
