"""One fixed-orientation item per bin: a minimal feasible 3D-BPP seed."""


def solve(instance):
    """Return [bin_index, x, y, z] for each item in input order."""
    return [[index, 0, 0, 0] for index in range(len(instance["items"]))]
