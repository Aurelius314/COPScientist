import numpy as np
import torch
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.distributions import Categorical
import seed_heuristic as solution_module
from heuristic_utils import emit_heuristic_result, parse_eval_args

heuristics = solution_module.heuristics

# =====ACO class=====
class ACO():
    def __init__(self,  # constraints are set to 1 after normalize weight 
                 prize,  # shape [n,]
                 weight, # shape [n, m]
                 heuristic,
                 n_ants=30, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 device='cpu'
                 ):
        self.n, self.m = weight.shape

        self.prize = prize
        self.weight = weight
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        self.pheromone = torch.ones(size=(self.n+1,), device=device)

        # Fidanova S. Hybrid ant colony optimization algorithm for multiple knapsack problem
        # self.heuristic = prize / self.weight.sum(dim=1) if heuristic is None else heuristic
        self.heuristic = heuristic
        # Leguizamon G, Michalewicz Z. A New Version of Ant System for Subset Problems
        self.Q = 1 / self.prize.sum()

        self.alltime_best_sol = None
        self.alltime_best_obj = 0
        self.device = device
        self.add_dummy_node()
        
    def add_dummy_node(self):
        self.prize = torch.cat((self.prize, torch.tensor([0.], device=self.device))) # (n+1,)
        self.weight = torch.cat((self.weight, torch.zeros((1, self.m), device=self.device)), dim=0) # (n+1, m)
        self.heuristic = torch.cat((self.heuristic, torch.tensor([1e-8], device=self.device))) # (n+1)

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            sols = self.gen_sol() # (n_ants, max_horizon)
            objs = self.gen_sol_obj(sols)             # (n_ants,)
            sols = sols.T
            best_obj, best_idx = objs.max(dim=0)
            if best_obj > self.alltime_best_obj:
                self.alltime_best_obj = best_obj
                self.alltime_best_sol = sols[best_idx]
            self.update_pheronome(sols, objs, best_obj.item(), best_idx.item())
        return self.alltime_best_obj, self.alltime_best_sol

    @torch.no_grad()
    def update_pheronome(self, sols, objs, best_obj, best_idx):
        self.pheromone = self.pheromone * self.decay 
        for i in range(self.n_ants):
            sol = sols[i]
            obj = objs[i]
            self.pheromone[sol] += self.Q * obj

    @torch.no_grad()
    def gen_sol_obj(self, solutions):
        '''
        Args:
            solutions: (n_ants, max_horizon)
        Return:
            obj: (n_ants,)
        '''
        return self.prize[solutions.T].sum(dim=1) # (n_ants,)

    def gen_sol(self):
        '''
        Solution contruction for all ants
        '''
        solutions = [] # solutions[i] is the i-th picked item for all ants
        knapsack = torch.zeros(size=(self.n_ants, self.m), device=self.device)  # used capacity
        mask = torch.ones(size=(self.n_ants, self.n+1), device=self.device)
        dummy_mask = torch.ones(size=(self.n_ants, self.n+1), device=self.device)
        dummy_mask[:, -1] = 0
        
        mask, knapsack = self.update_knapsack(mask, knapsack, new_item=None)
        dummy_mask = self.update_dummy_state(mask, dummy_mask)
        done = self.check_done(mask)
        while not done:
            items = self.pick_item(mask, dummy_mask)
            solutions.append(items)
            mask, knapsack = self.update_knapsack(mask, knapsack, items)
            dummy_mask = self.update_dummy_state(mask, dummy_mask)
            done = self.check_done(mask)
        return torch.stack(solutions)
    
    def pick_item(self, mask, dummy_mask):
        phe = self.pheromone.unsqueeze(0).repeat(self.n_ants, 1)
        heu = self.heuristic.unsqueeze(0).repeat(self.n_ants, 1)
        dist = ((phe ** self.alpha) * (heu ** self.beta) * mask * dummy_mask) # (n_ants, n+1)
        dist = Categorical(dist)
        item = dist.sample()
        return item # (n_ants,)
    
    def check_done(self, mask):
        # is mask all zero except for the dummy node?
        return (mask[:, :-1] == 0).all()
    
    def update_dummy_state(self, mask, dummy_mask):
        finished = (mask[: ,:-1] == 0).all(dim=1)
        dummy_mask[finished] = 1
        return dummy_mask
    
    def update_knapsack(self, mask, knapsack, new_item):
        '''
        Args:
            mask: (n_ants, n+1)
            knapsack: (n_ants, m)
            new_item: (n_ants)
        '''
        if new_item is not None:
            mask[torch.arange(self.n_ants), new_item] = 0
            knapsack += self.weight[new_item] # (n_ants, m)
        for ant_idx in range(self.n_ants):
            candidates = torch.nonzero(mask[ant_idx]) # (x, 1)
            if len(candidates) > 1:
                candidates.squeeze_()
                test_knapsack = knapsack[ant_idx].unsqueeze(0).repeat(len(candidates), 1) # (x, m)
                new_knapsack = test_knapsack + self.weight[candidates] # (x, m)
                infeasible_idx = candidates[(new_knapsack > 1).any(dim=1)]
                mask[ant_idx, infeasible_idx] = 0
        mask[:, -1] = 1
        return mask, knapsack

# =======Define evaluation function=====
N_ITERATIONS = 50
N_ANTS = 10

def solve(prize: np.ndarray, weight: np.ndarray):
    n, m = weight.shape
    heu = heuristics(prize.copy(), weight.copy()) + 1e-9
    assert heu.shape == (n,)
    heu[heu < 1e-9] = 1e-9
    aco = ACO(torch.from_numpy(prize), torch.from_numpy(weight), torch.from_numpy(heu), N_ANTS)
    obj, _ = aco.run(N_ITERATIONS)
    return obj

if __name__ == "__main__":
    import traceback

    args = parse_eval_args(100)
    try:
        basepath = Path(__file__).resolve().parent / "dataset"
        sizes = [args.problem_size] if args.mode == "train" else [100, 300, 500]
        metrics = {}
        for size in sizes:
            dataset_path = basepath / f"{args.mode}{size}_dataset.npz"
            with np.load(dataset_path) as dataset:
                prizes, weights = dataset["prizes"], dataset["weights"]
            print(f"[*] Evaluating {dataset_path} with {len(prizes)} instances.")
            values = [float(solve(prize, weight)) for prize, weight in zip(prizes, weights)]
            metrics[size] = float(np.mean(values))
            print(f"[*] Average for {size}: {metrics[size]}")
        emit_heuristic_result(metrics)
    except Exception as exc:
        print("__SANDBOX_ERROR__:")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc
