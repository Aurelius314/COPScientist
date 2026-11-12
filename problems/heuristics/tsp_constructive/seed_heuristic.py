"""Constructive TSP heuristic: bounded 2-opt + Or-opt commit on a nearest chain.

Mechanism
---------
On every construction step a provisional open path P = [current_node, <all
unvisited nodes>, destination] is seeded by a deterministic nearest-chain walk
(each step appends the closest still-unused node; O(n^2) one-pass construction).

The seeded path is then improved in two bounded refinement phases before the
successor of ``current_node`` is committed:

- Phase A (bounded 2-opt, champion style): at most MAX_PASSES=20 first-
  improvement passes over the per-node candidate lists, applying each reversal
  that strictly reduces the *open-path* length (relative decrease eps_rel =
  1e-9), with a hard MAX_MOVES=500 move cap and deterministic tie-breaking.

- Phase B (bounded Or-opt, new primitive): 2-opt reversals cannot change the
  cyclic order of any segment, so residual bulge/detour patterns near the
  committed edge persist; Phase B relocates length-L segments (L in {1,2,3})
  to legal candidate positions (j < s-1 or j > s+L), earliest-improvement
  deterministic scan, strict relative decrease, MAX_PASSES_OR=10 passes and
  MAX_MOVES_OR=500 moves. The two neighbourhoods are complementary in classic
  local-search practice (Lin-Kernighan style iterative improvement).

Candidate lists are region-informed, exactly as in the champion
``cluster_aware_bounded_two_opt``: every path node probes its k=12 raw nearest
neighbours inside the path UNION up to c=12 members of the same gap-seeded
cluster of the remaining set (primitives p5/p6/p7: nearest-neighbour gap,
seed selection, cluster assignment).

Scale-awareness: k=12, c=12 and all pass/move caps are fixed iteration limits,
so no score weight needs retuning with N or the coordinate scale; every
acceptance test is a plain Euclidean distance ratio eps_rel.

Feasibility at N=500: per call the nearest-chain seed walk and the region
computation (n x n gap submatrix) are O(n^2); each refinement phase is at most
(passes * candidates * n) = O(n) with fixed caps, and the move caps bound the
path updates, so a call is O(n^2) time and O(n^2) memory; a full tour of 500
calls is O(n^3) ~ 1.25e8 element operations, within numpy capacity.
Deterministic, no restarts, no rollout, no hidden state.
"""

import numpy as np

# Dimensionless weight balancing nearness (D[cur,j]/s) vs deferral cost (ic/s).
W_PINCH = 1.0


PRIMITIVE_PROVENANCE = {
    # Added primitive (operation "add" on slot _path_pinch_costs()): the bounded
    # 2-opt loop and the region-informed candidate-list construction are adopted
    # from the champion p10 (bounded_two_opt_commit / cluster_aware_bounded_two_opt)
    # and p5/p6/p7 (gap-seeded region structure); the Or-opt phase is new.
    "oropt_plus_twoopt_commit": ["p10_bounded_two_opt_commit", "p5_p6_p7_region_structure"],
}


def _nearest_chain_path(
    current_node: int,
    destination_node: int,
    unvisited_nodes: set,
    distance_matrix: np.ndarray,
) -> np.ndarray:
    """Greedy open path P = [current_node, ..., destination_node] over the
    remaining set: each step appends the closest still-unused node to the tail.

    O(n^2) time (n steps, each a vectorized O(n) argmin slice), O(n) memory.
    Deterministic: ties broken by the smallest node id (sorted candidates).
    """
    remaining = np.fromiter(sorted(unvisited_nodes), dtype=int)
    path = [current_node]
    tail = current_node
    while remaining.size:
        dists = distance_matrix[tail, remaining]
        j = int(np.argmin(dists))
        tail = int(remaining[j])
        path.append(tail)
        remaining = np.delete(remaining, j)
    path.append(destination_node)
    return np.asarray(path, dtype=int)


def _path_pinch_costs(
    path: np.ndarray,
    unvisited_nodes: set,
    distance_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-candidate pinch (deferral) cost ic(j) along path P.

    ic(j) = D[prev_j, j] + D[j, next_j] - D[prev_j, next_j], the detour the
    naive chain pays to insert j between its path neighbours.  Vectorized with
    advanced indexing: O(n) time, O(n) memory.
    """
    node_ids = np.fromiter(sorted(unvisited_nodes), dtype=int)
    pos = np.empty(int(distance_matrix.shape[0]), dtype=int)
    pos[path] = np.arange(path.size)
    idx = pos[node_ids]
    prev = path[idx - 1]
    nxt = path[idx + 1]
    pinch = (
        distance_matrix[prev, node_ids]
        + distance_matrix[node_ids, nxt]
        - distance_matrix[prev, nxt]
    )
    return node_ids, pinch


def _nearest_neighbour_gaps(unvisited_nodes, distance_matrix):
    """Step 1: nearest-neighbour gap g(j) = min_{k in U, k != j} D[j, k] per node."""
    u = np.fromiter(sorted(unvisited_nodes), dtype=int)
    sub = distance_matrix[np.ix_(u, u)]
    np.fill_diagonal(sub, np.inf)
    return u, sub.min(axis=1)


def _select_seeds(u, gaps):
    """Step 2: the ceil(sqrt(n)) nodes with the largest gaps become cluster seeds."""
    k = int(np.ceil(np.sqrt(u.size)))
    seed_idx = np.argpartition(-gaps, k - 1)[:k]
    return u[seed_idx]


def _assign_clusters(u, seeds, distance_matrix):
    """Step 3: assign every unvisited node to its nearest seed (a cluster id)."""
    return distance_matrix[np.ix_(u, seeds)].argmin(axis=1)


def oropt_plus_twoopt_commit(current_node, destination_node, unvisited_nodes, distance_matrix):
    """New primitive (experiment 1.1): bounded 2-opt + Or-opt commit.

    Commits the successor of ``current_node`` on an open path P over the whole
    remaining set that has been refined by two complementary, iteration-capped
    local-search phases:

    (1) Seed P = [current_node, <all unvisited in deterministic nearest-chain
        order>, destination] -- each step appends the closest still-unused node
        (O(n^2) one-pass construction, same walk as the champion commits).
    (2) Per-node candidate lists: for every path node the k=12 raw nearest
        neighbours inside the path UNION up to c=12 members of the same
        gap-seeded cluster (p5/p6/p7 over the remaining set); the union is
        bounded by 24 members regardless of n.
    (3) Phase A -- bounded 2-opt exactly in the champion style: MAX_PASSES=20
        first-improvement passes, strict relative decrease (delta < -eps_rel *
        removed edges with eps_rel = 1e-9), hard MAX_MOVES=500 move cap,
        deterministic scan order.
    (4) Phase B -- bounded Or-opt (node relocation): for segment lengths L in
        {1, 2, 3} each length-L segment starting at s is moved to every
        candidate position j of its head's candidate list with a legal
        insertion (j < s-1 or j > s+L, j+1 <= m-1), earliest-improvement
        deterministic scan, strict relative decrease, at most MAX_PASSES_OR=10
        passes and MAX_MOVES_OR=500 accepted moves.  This removes bulge/detour
        patterns that 2-opt reversals cannot fix because reversals preserve
        cyclic segment order.
    (5) Return path[1], the successor of ``current_node`` in the refined path.

    Iteration caps keep the per-call cost O(n^2) time / O(n^2) memory (seed
    walk + region submatrix dominate; each phase is O(n * (k + c) * passes)
    with fixed caps); a full tour is O(n^3) at N=500.  Deterministic, no
    restarts, no full tour completion, no hidden state.
    """
    k = 12
    c = 12
    max_passes = 20
    max_moves = 500
    eps_rel = 1e-9
    max_passes_or = 10
    max_moves_or = 500

    # Region structure over the remaining set (primitives p5/p6/p7).
    u, gaps = _nearest_neighbour_gaps(unvisited_nodes, distance_matrix)
    seeds = _select_seeds(u, gaps)
    cluster_id = _assign_clusters(u, seeds, distance_matrix)
    cluster_of = {int(node): int(cid) for node, cid in zip(u, cluster_id)}

    def _region_of(node):
        """Cluster id of a path node; current/destination map to their nearest seed."""
        cid = cluster_of.get(int(node))
        if cid is None:
            cid = int(np.argmin(distance_matrix[node, seeds]))
        return cid

    # (1) deterministic nearest-chain seed path.
    remaining = u.tolist()
    path = [current_node]
    pos = current_node
    while remaining:
        dists = distance_matrix[pos, remaining]
        idx = int(np.argmin(dists))
        pos = remaining.pop(idx)
        path.append(pos)
    path.append(destination_node)

    m = len(path)
    nodes = np.asarray(path, dtype=int)
    sub = distance_matrix[np.ix_(nodes, nodes)]
    np.fill_diagonal(sub, np.inf)

    # Same-region membership lists, grouped once in path order (deterministic).
    by_region = {}
    for node in path:
        by_region.setdefault(_region_of(node), []).append(int(node))

    # (2) per-node candidate lists: {k raw NN} UNION {up to c same-cluster members}.
    kk = min(k, m - 1)
    if kk > 0:
        part = np.argpartition(sub, kk - 1, axis=1)[:, :kk]
    else:
        part = None
    cand_of = {}
    for row in range(m):
        node = int(nodes[row])
        raw = [int(nodes[t]) for t in part[row]] if part is not None else []
        combined = list(raw)
        seen = set(raw)
        for w in by_region[_region_of(node)]:
            if w != node and w not in seen:
                seen.add(w)
                combined.append(w)
                if len(combined) >= k + c:
                    break
        cand_of[node] = combined

    pos_of = {node: i for i, node in enumerate(path)}
    moves = 0

    # (3) Phase A: bounded 2-opt (champion loop, first-improvement, strict
    #     relative decrease).  Reversals do not touch path[0] or path[m-1].
    for _ in range(max_passes):
        improved_any = False
        i = 0
        while i < m - 2 and moves < max_moves:
            a, b = path[i], path[i + 1]
            improved = False
            for q in cand_of[b]:
                j = pos_of[q]
                if j <= i + 1 or j >= m - 1:
                    continue
                c2, d = path[j], path[j + 1]
                ab = distance_matrix[a, b]
                cd = distance_matrix[c2, d]
                delta = distance_matrix[a, c2] + distance_matrix[b, d] - ab - cd
                if delta < -eps_rel * (ab + cd):
                    path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                    for t in range(i + 1, j + 1):
                        pos_of[path[t]] = t
                    improved = True
                    improved_any = True
                    moves += 1
                    break
            if not improved:
                i += 1
        if not improved_any or moves >= max_moves:
            break

    # (4) Phase B: bounded Or-opt (node relocation), earliest-improvement
    #     deterministic scan; each pass restarts from L=1 after each accepted move.
    moves = 0
    for _ in range(max_passes_or):
        changed = False
        improved = False
        for seg_len in (1, 2, 3):
            s = 1
            while s + seg_len <= m - 1 and moves < max_moves_or:
                head = path[s]
                prev = path[s - 1]
                tail = path[s + seg_len - 1]
                nxt = path[s + seg_len]
                for q in cand_of[head]:
                    j = pos_of[q]
                    if j < m - 1 and (j < s - 1 or j > s + seg_len):
                        c3 = path[j + 1]
                        removed = (
                            distance_matrix[prev, head]
                            + distance_matrix[tail, nxt]
                            + distance_matrix[path[j], c3]
                        )
                        delta = (
                            distance_matrix[prev, nxt]
                            + distance_matrix[path[j], head]
                            + distance_matrix[tail, c3]
                            - removed
                        )
                        if delta < -eps_rel * removed:
                            seg = path[s:s + seg_len]
                            del path[s:s + seg_len]
                            # Insertion index must be recomputed after the
                            # deletion: a q that sat after the removed segment
                            # has shifted left by seg_len.  (A raw
                            # pos_of[q] + 1 could land past the end and push
                            # the destination into the interior.)
                            q_new = pos_of[q]
                            if q_new > s:
                                q_new -= seg_len
                            ins = q_new + 1
                            path[ins:ins] = seg
                            for t in range(m):
                                pos_of[path[t]] = t
                            moves += 1
                            changed = True
                            improved = True
                            break
                if improved:
                    break
                s += 1
            if improved:
                break  # restart the deterministic scan from L=1 on the new path
        if not changed or moves >= max_moves_or:
            break

    return int(path[1])


def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    """Select the next node to visit: bounded 2-opt + Or-opt commit
    (``oropt_plus_twoopt_commit``) -- region-informed candidate lists (k=12 raw
    nearest neighbours UNION up to c=12 same gap-seeded cluster members) inside
    a bounded 2-opt phase (MAX_PASSES=20, MAX_MOVES=500) followed by a bounded
    Or-opt relocation phase (MAX_PASSES_OR=10, MAX_MOVES_OR=500).

    Falls back to the smallest remaining index (legacy primitive p1) only in
    the degenerate single-node case, where every rule coincides.  O(n^2) per
    call, O(n^2) memory, deterministic, no mutation of the inputs and no hidden
    state between calls.
    """
    if len(unvisited_nodes) <= 1:
        return min(unvisited_nodes)
    return oropt_plus_twoopt_commit(current_node, destination_node, unvisited_nodes, distance_matrix)
