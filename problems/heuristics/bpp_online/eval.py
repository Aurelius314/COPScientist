"""
Adapted from https://github.com/google-deepmind/funsearch
"""

import numpy as np
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heuristic_utils import parse_eval_args

from seed_solution import priority

def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]

def online_binpack(
        items: tuple[float, ...], bins: np.ndarray
) -> tuple[list[list[float, ...], ...], np.ndarray]:
    """Performs online binpacking of `items` into `bins`."""
    # Track which items are added to each bin.
    packing = [[] for _ in bins]
    # Add items to bins.
    for item in items:
        # Extract bins that have sufficient space to fit item.
        valid_bin_indices = get_valid_bin_indices(item, bins)
        # Score each bin based on heuristic.
        priorities = priority(item, bins[valid_bin_indices])
        # Add item to bin with highest priority.
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    # Remove unused bins from packing.
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins

def evaluate(instances: dict) -> tuple[float, dict[str, float]]:
    """Return the mean bin count and the bin count for each instance."""
    metrics = {}
    # Perform online binpacking for each instance.
    for name in instances:
        if name == 'l1_bound':
            continue
        instance = instances[name]
        capacity = instance['capacity']
        items = instance['items']
        items = np.array(items) if isinstance(items, list) else items
        # Create num_items bins so there will always be space for all items,
        # regardless of packing order. Array has shape (num_items,).
        bins = np.array([capacity for _ in range(instance['num_items'])])
        # Pack items into bins and return remaining capacity in bins_packed, which
        # has shape (num_items,).
        _, bins_packed = online_binpack(items.astype(float), bins)
        # If remaining capacity in a bin is equal to initial capacity, then it is
        # unused. Count number of used bins.
        metrics[name] = float((bins_packed != capacity).sum())
    if not metrics:
        raise ValueError("Dataset contains no bin-packing instances")
    return float(np.mean(list(metrics.values()))), metrics


def get_feature(metrics: dict[str, float]) -> tuple[int, ...]:
    """Discretize the bin count for each instance in dataset order."""
    return tuple(int(value) for value in metrics.values())

def is_valid_packing(
        packing: list[list[float, ...], ...], items: list[float], capacity: float
) -> bool:
    """Returns whether `packing` is valid.

    Returns whether `packing` is a valid packing of `items` into bins of size
    `capacity`.

    Args:
        packing: Packing of items into bins. List of bins, where each bin contains
        a list of items packed into that bin.
        items: List of item sizes.
        capacity: Capacity of each bin.
    """
    # Check that items in packing are exactly the same as list of input items.
    packed_items = sum(packing, [])  # Join items in each bin into a single list.
    if sorted(packed_items) != sorted(items):
        return False

    # Check that each bin contains less than `capacity` items .
    for bin_items in packing:
        if sum(bin_items) > capacity:
            return False

    return True

if __name__ == "__main__":
    import traceback

    args = parse_eval_args(100)
    try:
        basepath = Path(__file__).resolve().parent / "dataset"
        dataset_path = basepath / f"weibull_5k_{args.mode}.pickle"
        with dataset_path.open("rb") as stream:
            dataset = pickle.load(stream)
        score, metrics = evaluate(dataset)
        l1_bound = float(dataset["l1_bound"])
        excess = (score - l1_bound) / l1_bound * 100
        features = get_feature(metrics)
        print(f"[*] Evaluated {dataset_path}")
        print(f"[*] Average bins: {score}; L1 bound: {l1_bound}; excess: {excess:.2f}%")
        print("__SANDBOX_RESULT__")
        print("__METRICS_START__")
        print(repr(metrics))
        print("__METRICS_END__")
        print("__FEATURES_START__")
        print(repr(features))
        print("__FEATURES_END__")
        print("__SCORE_START__")
        print(repr(score))
        print("__SCORE_END__")
        print("__SANDBOX_SUCCESS__")
    except Exception as exc:
        print("__SANDBOX_ERROR__:")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc
