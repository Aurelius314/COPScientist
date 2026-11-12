"""Feasible baseline: one fixed-orientation item per bin."""


def solve(instance):
    return [[index, 0, 0, 0] for index in range(len(instance["items"]))]
