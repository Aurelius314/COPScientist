"""Simple job sequence baseline; evaluator schedules each operation as early as possible."""


def solve(instance):
    counts = [0] * instance["jobs"]
    for operation in instance["operations"]:
        counts[operation["job"]] += 1
    sequence = []
    for index in range(max(counts)):
        sequence.extend(job for job, count in enumerate(counts) if index < count)
    return sequence
