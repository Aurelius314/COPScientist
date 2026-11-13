import numpy as np


def select_next_item(remaining_capacity, weights, values):
    """Return the index of the next item to pack.

    Seed heuristic: always pick item 0 (the trivial baseline).
    """
    next_item = 0
    return next_item