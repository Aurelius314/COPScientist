import os
import sys
import numpy as np
from scipy.spatial import distance_matrix
import logging
import inspect
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.distributions import Categorical
import seed_heuristic as solution_module
from heuristic_utils import parse_eval_args

heuristics = solution_module.heuristics

# =====ACO class=====
class ACO():
    def __init__(self,  # 0: depot
                 distances, # (n, n) distance matrix between all nodes
                 demand,   # (n, ) demand at each node (0 for depot)
                 heuristic, # (n, n) heuristic matrix guiding ant movement
                 capacity,  # vehicle capacity constraint
                 n_ants=30,  # number of ants in colony
                 decay=0.9,  # pheromone evaporation rate
                 alpha=1,    # pheromone importance factor
                 beta=1,     # heuristic importance factor
                 device='cpu',  # computation device
                 ):
        self.problem_size = len(distances)  # number of nodes including depot
        self.distances = torch.tensor(distances, device=device) if not isinstance(distances, torch.Tensor) else distances
        self.demand = torch.tensor(demand, device=device) if not isinstance(demand, torch.Tensor) else demand
        self.capacity = capacity

        self.n_ants = n_ants
        self.decay = decay  # pheromone evaporation: τ = τ * decay
        self.alpha = alpha  # controls pheromone influence: τ^α
        self.beta = beta    # controls heuristic influence: η^β

        self.pheromone = torch.ones_like(self.distances)  # initial pheromone matrix
        self.heuristic = torch.tensor(heuristic, device=device) if not isinstance(heuristic, torch.Tensor) else heuristic

        self.shortest_path = None  # best solution found
        self.lowest_cost = float('inf')  # cost of best solution

        self.device = device

    @torch.no_grad()
    def run(self, n_iterations):
        """Main ACO loop: run for n_iterations"""
        for _ in range(n_iterations):
            paths = self.gen_path()  # generate paths for all ants
            costs = self.gen_path_costs(paths)  # compute total distance for each ant

            best_cost, best_idx = costs.min(dim=0)  # find best ant in this iteration
            if best_cost < self.lowest_cost:  # update global best if improved
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost

            self.update_pheronome(paths, costs)  # update pheromone trails

        return self.lowest_cost  # return best cost found

    @torch.no_grad()
    def update_pheronome(self, paths, costs):
        '''
        Update pheromone trails using ant solutions.
        Pheromone update rule: τ_ij = τ_ij * decay + Σ(Δτ_ij^k) where Δτ_ij^k = Q/L_k

        Args:
            paths: torch tensor with shape (problem_size, n_ants) - complete paths for all ants
            costs: torch tensor with shape (n_ants,) - total distance for each ant
        '''
        self.pheromone = self.pheromone * self.decay  # evaporation: τ = τ * ρ
        for i in range(self.n_ants):
            path = paths[:, i]  # path for ant i
            cost = costs[i]     # total distance for ant i
            # Add pheromone to edges used by this ant: Δτ = Q/L (Q=1 here)
            # path[:-1] gives current nodes, torch.roll(path, shifts=-1)[:-1] gives next nodes
            self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0/cost
        self.pheromone[self.pheromone < 1e-10] = 1e-10  # prevent pheromone from going to zero

    @torch.no_grad()
    def gen_path_costs(self, paths):
        """Compute total distance for each ant's path"""
        u = paths.permute(1, 0) # shape: (n_ants, max_seq_len) - transpose for easier indexing
        v = torch.roll(u, shifts=-1, dims=1)  # shift to get next node in sequence
        # Sum distances between consecutive nodes (excluding last to first wrap-around)
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def gen_path(self):
        """Generate complete paths for all ants using constructive heuristic"""
        actions = torch.zeros((self.n_ants,), dtype=torch.long, device=self.device)  # all ants start at depot (node 0)
        visit_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)  # 1=unvisited, 0=visited
        visit_mask = self.update_visit_mask(visit_mask, actions)  # mark depot as visited
        used_capacity = torch.zeros(size=(self.n_ants,), device=self.device)  # current load for each ant

        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)  # update capacity constraints

        paths_list = [actions]  # paths_list[i] contains the ith move for all ants

        done = self.check_done(visit_mask, actions)
        while not done:
            actions = self.pick_move(actions, visit_mask, capacity_mask)  # probabilistic node selection
            paths_list.append(actions)  # record move
            visit_mask = self.update_visit_mask(visit_mask, actions)  # update visited nodes
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)  # update capacity
            done = self.check_done(visit_mask, actions)  # check termination

        return torch.stack(paths_list)  # shape: (seq_len, n_ants)

    def pick_move(self, prev, visit_mask, capacity_mask):
        """Probabilistic node selection using transition probability: p_ij ∝ τ_ij^α * η_ij^β"""
        pheromone = self.pheromone[prev]  # shape: (n_ants, p_size) - pheromone on edges from current nodes
        heuristic = self.heuristic[prev]  # shape: (n_ants, p_size) - heuristic values from current nodes
        # Transition probability: p_ij = (τ_ij^α * η_ij^β) / Σ(τ_ik^α * η_ik^β)
        # Masked by visit_mask (unvisited nodes) and capacity_mask (feasible nodes)
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * visit_mask * capacity_mask)  # shape: (n_ants, p_size)
        dist = Categorical(dist)  # create categorical distribution
        actions = dist.sample()  # shape: (n_ants,) - sample next node for each ant
        return actions

    def update_visit_mask(self, visit_mask, actions):
        """Update mask of unvisited nodes after moving to new nodes"""
        visit_mask[torch.arange(self.n_ants, device=self.device), actions] = 0  # mark new nodes as visited
        visit_mask[:, 0] = 1  # depot can always be revisited (for returning/starting new route)
        # Exception: if ant returns to depot AND still has unvisited customers, don't allow immediate return
        # This prevents depot-depot cycles when work remains
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0
        return visit_mask

    def update_capacity_mask(self, cur_nodes, used_capacity):
        '''
        Update vehicle capacity constraints and create mask of feasible next nodes.

        Args:
            cur_nodes: shape (n_ants, ) - current node for each ant
            used_capacity: shape (n_ants, ) - current load for each ant

        Returns:
            used_capacity: updated capacity after visiting cur_nodes
            capacity_mask: mask where 1=feasible (demand ≤ remaining capacity), 0=infeasible
        '''
        capacity_mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        # update capacity: reset to 0 when returning to depot, add demand of current node
        used_capacity[cur_nodes==0] = 0  # reset load when returning to depot
        used_capacity = used_capacity + self.demand[cur_nodes]  # add demand of current node

        # update capacity_mask: mask out nodes whose demand exceeds remaining capacity
        remaining_capacity = self.capacity - used_capacity  # (n_ants,) - remaining capacity for each ant
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size)  # (n_ants, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.n_ants, 1)  # (n_ants, p_size) - demand of all nodes
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0  # mask infeasible nodes

        return used_capacity, capacity_mask

    def check_done(self, visit_mask, actions):
        """Check termination condition: all customers visited and all ants at depot"""
        # All customers (nodes 1..n) visited AND all ants currently at depot (node 0)
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()

# =====Evaluation function=====
N_ITERATIONS = 100
N_ANTS = 30
CAPACITY = 50

def solve(node_pos, demand):
    dist_mat = distance_matrix(node_pos, node_pos)
    dist_mat[np.diag_indices_from(dist_mat)] = 1 # set diagonal to a large number
    if len(inspect.getfullargspec(heuristics).args) == 4:
        heu = heuristics(dist_mat.copy(), node_pos.copy(), demand.copy(), CAPACITY) + 1e-9
    elif len(inspect.getfullargspec(heuristics).args) == 2:
        heu = heuristics(dist_mat.copy(), demand / CAPACITY) + 1e-9
    heu[heu < 1e-9] = 1e-9
    aco = ACO(dist_mat, demand, heu, CAPACITY, n_ants=N_ANTS)
    obj = aco.run(N_ITERATIONS)
    return obj


def get_feature(metrics: dict[int, float]) -> tuple[int, ...]:
    """Discretize the mean objective for each problem size."""
    return tuple(int(metrics[size]) for size in sorted(metrics))


def get_score(metrics: dict[int, float]) -> float:
    """Average the unrounded mean objectives across problem sizes."""
    if not metrics:
        raise ValueError("Evaluation produced no metrics")
    return sum(metrics.values()) / len(metrics)

if __name__ == "__main__":
    import traceback

    args = parse_eval_args(50)
    try:
        basepath = Path(__file__).resolve().parent / "dataset"
        sizes = [args.problem_size] if args.mode == "train" else [20, 50, 100]
        metrics = {}
        for size in sizes:
            dataset_path = basepath / f"{args.mode}{size}_dataset.npy"
            dataset = np.load(dataset_path)
            demands, positions = dataset[:, :, 0], dataset[:, :, 1:]
            print(f"[*] Evaluating {dataset_path} with {len(dataset)} instances.")
            values = [float(solve(pos, demand)) for pos, demand in zip(positions, demands)]
            metrics[size] = float(np.mean(values))
            print(f"[*] Average for {size}: {metrics[size]}")
        features = get_feature(metrics)
        score = get_score(metrics)
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
