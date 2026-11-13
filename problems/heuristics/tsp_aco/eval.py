from os import path
import sys
import numpy as np
from scipy.spatial import distance_matrix
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import traceback
import torch
from torch.distributions import Categorical
import seed_solution as solution_module
from heuristic_utils import emit_heuristic_result, parse_eval_args

heuristics = solution_module.heuristics

# ======Define ACO class=====
class ACO():
    def __init__(self,
                 distances,
                 heuristic,
                 n_ants=30,
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 device='cpu'
                 ):
        """
        Ant Colony Optimization (ACO) for Traveling Salesman Problem (TSP)

        Args:
            distances: Distance matrix between cities (n x n)
            heuristic: Heuristic matrix (usually 1/distance) for attractiveness
            n_ants: Number of ants (agents) per iteration
            decay: Pheromone evaporation rate (0-1)
            alpha: Pheromone importance weight
            beta: Heuristic importance weight
            device: 'cpu' or 'cuda' for PyTorch tensors
        """
        self.problem_size = len(distances)
        self.distances  = torch.tensor(distances, device=device) if not isinstance(distances, torch.Tensor) else distances
        self.n_ants = n_ants
        self.decay = decay  # Pheromone evaporation rate
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance

        # Initialize pheromone matrix with equal values on all edges
        self.pheromone = torch.ones_like(self.distances)  # Pheromone level on each edge between cities; shape (n, n)
        self.heuristic = torch.tensor(heuristic, device=device) if not isinstance(heuristic, torch.Tensor) else heuristic  # Attractiveness of each edge; shape (n, n)

        # Track best solution found
        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device

    @torch.no_grad()
    def run(self, n_iterations):
        """
        Main ACO loop: run for n_iterations

        Returns:
            Best tour cost found
        """
        for _ in range(n_iterations):
            # 1. Tour construction: all ants build tours
            # `paths: Shape (problem_size, n_ants)
            # Each column is one ant's complete tour (sequence of city indices); paths[:, i] = tour of ant i
            paths = self.gen_path(require_prob=False)
            
            # 2. Evaluate tour costs
            # `costs``: Shape (n_ants,); costs[i] = cost of ant i's tour
            costs = self.gen_path_costs(paths)

            # 3. Update global best solution
            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost

            # 4. Pheromone update (evaporation + deposition)
            self.update_pheronome(paths, costs)

        return self.lowest_cost  # The method returns the lowest cost found across all N_ITERATIONS.
       
    @torch.no_grad()
    def update_pheronome(self, paths, costs):
        '''
        Update pheromone matrix: evaporation + ant deposition

        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        # 1. Evaporation: reduce all pheromone values
        self.pheromone = self.pheromone * self.decay

        # 2. Deposition: each ant deposits pheromone on edges it used
        for i in range(self.n_ants):
            path = paths[:, i]
            cost = costs[i]
            # Deposit pheromone proportional to 1/cost (better tours deposit more)
            self.pheromone[path, torch.roll(path, shifts=1)] += 1.0/cost
            self.pheromone[torch.roll(path, shifts=1), path] += 1.0/cost  # Symmetric for undirected graph

    @torch.no_grad()
    def gen_path_costs(self, paths):
        '''
        Calculate total distance for each ant's tour

        Args:
            paths: torch tensor with shape (problem_size, n_ants)
        Returns:
            Lengths of paths: torch tensor with shape (n_ants,)
        '''
        assert paths.shape == (self.problem_size, self.n_ants)
        u = paths.T  # shape: (n_ants, problem_size) - current cities
        v = torch.roll(u, shifts=1, dims=1)  # shape: (n_ants, problem_size) - next cities (wraps around)
        assert (self.distances[u, v] > 0).all()
        # Sum distances along each ant's tour
        return torch.sum(self.distances[u, v], dim=1)

    def gen_path(self, require_prob=False):
        '''
        Tour construction for all ants (parallel construction)

        Returns:
            paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        # Random starting cities for each ant
        start = torch.randint(low=0, high=self.problem_size, size=(self.n_ants,), device=self.device)
        # Mask to track visited cities (0 = visited, 1 = available)
        mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        mask[torch.arange(self.n_ants, device=self.device), start] = 0

        paths_list = []  # paths_list[i] is the ith move (tensor) for all ants
        paths_list.append(start)

        log_probs_list = []  # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions

        prev = start
        for _ in range(self.problem_size-1):
            # Each ant chooses next city based on pheromone and heuristic
            actions, log_probs = self.pick_move(prev, mask, require_prob)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                mask = mask.clone()  # Clone mask when tracking probabilities
            prev = actions
            mask[torch.arange(self.n_ants, device=self.device), actions] = 0  # Mark new city as visited

        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)
        
    def pick_move(self, prev, mask, require_prob):
        '''
        Probabilistic selection of next city for each ant

        Args:
            prev: tensor with shape (n_ants,), previous nodes for all ants
            mask: bool tensor with shape (n_ants, p_size), masks (0) for the visited cities
        '''
        # Get pheromone and heuristic values for edges from current cities
        pheromone = self.pheromone[prev]  # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev]  # shape: (n_ants, p_size)

        # Probability distribution: P(i,j) ∝ [τ(i,j)^α] * [η(i,j)^β]
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * mask)  # shape: (n_ants, p_size)
        dist = Categorical(dist)

        # Sample next city for each ant
        actions = dist.sample()  # shape: (n_ants,)
        log_probs = dist.log_prob(actions) if require_prob else None  # shape: (n_ants,)
        return actions, log_probs


# =======Define evaluation function=====
N_ITERATIONS = 100
N_ANTS = 30

def solve(node_pos, *, n_iterations=N_ITERATIONS, n_ants=N_ANTS):
    dist_mat = distance_matrix(node_pos, node_pos)
    dist_mat[np.diag_indices_from(dist_mat)] = 1 # set diagonal to a large number
    heu = heuristics(dist_mat.copy()) + 1e-9
    heu[heu < 1e-9] = 1e-9
    aco = ACO(dist_mat, heu, n_ants=n_ants)
    obj = aco.run(n_iterations)
    return obj

if __name__ == "__main__":
    

    args = parse_eval_args(50, aco_options=True)
    try:
        basepath = Path(__file__).resolve().parent / "dataset"
        sizes = [args.problem_size] if args.mode == "train" else [20, 50, 100]
        metrics = {}
        for size in sizes:
            dataset_path = basepath / f"{args.mode}{size}_dataset.npy"
            instances = np.load(dataset_path)
            if args.max_instances is not None:
                instances = instances[:args.max_instances]
            print(f"[*] Evaluating {dataset_path} with {len(instances)} instances.")
            values = [float(solve(node_pos, n_iterations=args.iterations,
                                  n_ants=args.ants)) for node_pos in instances]
            metrics[size] = float(np.mean(values))
            print(f"[*] Average for {size}: {metrics[size]}")
        emit_heuristic_result(metrics)
    except Exception as exc:
        print("__SANDBOX_ERROR__:")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc
