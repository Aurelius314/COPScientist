"""MWKR dispatching rule: choose the job with the most work remaining.

Runarsson et al., "Pilot, Rollout and Monte Carlo Tree Search Methods for
Job Shop Scheduling", Algorithm 1. The evaluator schedules the returned
job sequence as early as possible.
"""


def solve(instance):
    remaining = [0] * instance["jobs"]
    for operation in instance["operations"]:
        remaining[operation["job"]] += operation["duration"]

    sequence = []
    next_step = [0] * instance["jobs"]
    operations = [[] for _ in next_step]
    for operation in instance["operations"]:
        operations[operation["job"]].append(operation)
    for job_ops in operations:
        job_ops.sort(key=lambda operation: operation["step"])

    for _ in range(len(instance["operations"])):
        job = max(
            (job for job, job_ops in enumerate(operations) if next_step[job] < len(job_ops)),
            key=lambda job: (remaining[job], -job),
        )
        operation = operations[job][next_step[job]]
        sequence.append(job)
        remaining[job] -= operation["duration"]
        next_step[job] += 1
    return sequence
