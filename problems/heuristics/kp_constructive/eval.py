import math
from os import path
import numpy as np
import sys
import argparse
from scipy.spatial import distance_matrix
import logging
import copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heuristic_utils import emit_heuristic_result, parse_eval_args

from seed_solution import select_next_item

def eval_heuristic(node_positions: np.ndarray, capacity) -> float:

    problem_size = node_positions.shape[0]
    # calculate distance matrix
    # set the starting node
    solution_value = 0.0
    remaining_capacity = copy.deepcopy(capacity)
    # init unvisited nodes
    weight = node_positions[:, 0]
    # remove the starting node
    value = node_positions[:, 1]
    # run the heuristic
    while weight.size and remaining_capacity + 1e-3 > min(weight):
        next_node = select_next_item(
            remaining_capacity=remaining_capacity,
            weights=weight,
            values=value
        )
        if 0 <= next_node < len(value) and remaining_capacity + 1e-3 > weight[next_node]:
            solution_value += value[next_node]
            remaining_capacity -= weight[next_node]
            weight = np.delete(weight, [next_node])
            value = np.delete(value, [next_node])
        else:
            print(remaining_capacity, weight)
            raise KeyError(f"Node {next_node} is illegal.")

    # calculate the length of the tour
    return solution_value

if __name__ == "__main__":
    import traceback

    args = parse_eval_args(100)
    try:
        basepath = Path(__file__).resolve().parent / "dataset"
        sizes = [args.problem_size] if args.mode == "train" else [50, 100, 200]
        metrics = {}
        for size in sizes:
            capacity = 12.5 if size == 50 else 25
            dataset_path = basepath / f"{args.mode}{size}_dataset.npy"
            instances = np.load(dataset_path)
            print(f"[*] Evaluating {dataset_path} with {len(instances)} instances.")
            values = [float(eval_heuristic(items, capacity)) for items in instances]
            metrics[size] = float(np.mean(values))
            print(f"[*] Average for {size}: {metrics[size]}")
        emit_heuristic_result(metrics)
    except Exception as exc:
        print("__SANDBOX_ERROR__:")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc
