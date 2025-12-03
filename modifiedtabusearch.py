import random, math
from typing import List, Tuple
import numpy as np
from numba import njit
from dataclasses import dataclass
from typing import List, Optional, Tuple
import itertools
import time

# ---------- energy ----------
@njit
def corr_energy_numba(a:list[int]) -> int:
    a = np.array(a)
    n = a.shape[0]
    m = n // 2
    e = 0
    for s in range(1, m+1):
        c = 0
        for i in range(n):
            c += a[i] * a[(i+s) % n]
        e += abs(c)
    return e


@njit
def correlation_vector(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    m = n // 2
    C = np.zeros(m, dtype=np.int64)
    for s in range(1, m+1):
        c = 0
        for i in range(n):
            c += a[i] * a[(i+s) % n]
        C[s-1] = c
    return C

@dataclass
class CWValidationResult:
    valid: bool
    weight: int
    target_weight: int
    sum: int
    target_sum_abs: int
    correlations: List[int]
    first_violation: Optional[str]


def validate_cw(a: List[int], k: int) -> CWValidationResult:
    n = len(a)
    weight = sum(1 for x in a if x != 0)
    target_weight = k * k
    s = sum(a)
    target_sum_abs = k

    m = n // 2   # ALWAYS floor(n/2)

    correlations = []
    first_violation = None

    # Check periodic autocorrelations for s = 1..m
    for shift in range(1, m + 1):
        c = sum(a[i] * a[(i + shift) % n] for i in range(n))
        correlations.append(c)
        if first_violation is None and c != 0:
            first_violation = f"C({shift}) = {c} != 0"

    # Weight / sum checks AFTER correlations
    if first_violation is None:
        if weight != target_weight:
            first_violation = f"weight {weight} != {target_weight}"
        elif abs(s) != target_sum_abs:
            first_violation = f"|sum| {abs(s)} != {k}"
        # optional: sum^2 check (redundant when |sum| checked)
        elif s * s != target_weight:
            first_violation = f"sum^2 {s*s} != k^2"

    return CWValidationResult(
        valid=(first_violation is None),
        weight=weight,
        target_weight=target_weight,
        sum=s,
        target_sum_abs=target_sum_abs,
        correlations=correlations,
        first_violation=first_violation
    )

def pick_divisor(n: int) -> int:
    nice_divs = [16, 12, 8, 6, 4, 3, 2]
    divs = [d for d in nice_divs if n % d == 0]
    return divs[0] if divs else 1

def coset_id(i: int, d: int) -> int:
    return i % d

def make_grouped_seed(n, k, d=None):
    """
    Make a grouped CW seed using residue classes (cosets) mod d.
    """
    weight = k*k
    if d is None:
        # pick a nice divisor of n
        for x in [16,12,8,6,4,3,2]:
            if n % x == 0:
                d = x
                break
        else:
            d = 1

    cosets = [[] for _ in range(d)]
    for i in range(n):
        cosets[i % d].append(i)

    # distribute weight across cosets evenly
    Wc = [weight//d]*d
    for i in range(weight % d):
        Wc[i] += 1

    a = [0]*n
    target_sum = k if random.random() < 0.5 else -k
    p = (weight + target_sum)//2
    q = weight - p

    for c in range(d):
        w = Wc[c]
        if w==0: continue

        chosen = random.sample(cosets[c], w)
        p_c = max(0, min(p, w))
        pluses = set(random.sample(chosen, p_c))
        p -= p_c

        for idx in chosen:
            a[idx] = 1 if idx in pluses else -1

    return a



def make_structured_cw_coset(n: int, k: int) -> List[int]:
    """
    Structured CW(n, k^2) seed:
      - exactly k^2 nonzeros
      - sum = ±k
      - support distributed across residue classes mod d (cosets)
    """
    weight = k * k
    if weight > n:
        raise ValueError("k^2 > n")

    # choose a "nice" divisor d of n (coset size)
    nice_divs = [16, 12, 8, 6, 4, 3, 2]
    divs = [d for d in nice_divs if n % d == 0]
    d = divs[0] if divs else 1  # if no nice divisor, fall back to 1 (no coset structure)

    # residue classes mod d
    cosets = [[] for _ in range(d)]
    for i in range(n):
        cosets[i % d].append(i)

    # distribute weight across cosets as evenly as possible
    base = weight // d
    rem  = weight % d
    coset_weights = [base] * d
    for c in range(rem):
        coset_weights[c] += 1

    # decide global sum ±k
    target_sum = k if random.random() < 0.5 else -k

    # total plus/minus counts
    # p + q = weight, p - q = target_sum
    if (weight + target_sum) % 2 != 0:
        target_sum = -target_sum  # fix parity if needed
    p_total = (weight + target_sum) // 2
    q_total = weight - p_total

    # distribute plus counts p_c across cosets in proportion to coset_weights
    plus_per_coset = [0] * d
    if weight > 0:
        # initial proportional assignment
        frac = [cw * p_total / weight for cw in coset_weights]
        plus_per_coset = [int(round(x)) for x in frac]
        # fix rounding to ensure sum equals p_total
        diff = sum(plus_per_coset) - p_total
        if diff != 0:
            # adjust randomly
            indices = list(range(d))
            random.shuffle(indices)
            step = 1 if diff > 0 else -1
            diff = abs(diff)
            for idx in indices:
                if diff == 0:
                    break
                # we can safely adjust as long as 0 <= plus_c <= coset_weights[c]
                new_val = plus_per_coset[idx] - step
                if 0 <= new_val <= coset_weights[idx]:
                    plus_per_coset[idx] = new_val
                    diff -= 1

    # now build sequence
    a = [0] * n
    p_left = p_total
    q_left = q_total

    for c in range(d):
        w_c = coset_weights[c]
        if w_c == 0:
            continue
        positions = cosets[c]

        # choose support positions in this coset
        support_c = random.sample(positions, w_c)

        # plus count for this coset, bounded by what's left
        p_c = min(plus_per_coset[c], p_left, w_c)
        q_c = w_c - p_c
        p_left -= p_c
        q_left -= q_c

        plus_positions = set(random.sample(support_c, p_c))
        for idx in support_c:
            a[idx] = 1 if idx in plus_positions else -1

    # sanity fix if plus/minus counts slightly off due to rounding
    # we only flip signs in +/- pairs to fix global sum
    s = sum(a)
    if abs(s) != abs(target_sum):
        # try to fix by flipping sign pairs (+1,-1) -> (-1,+1)
        pos = [i for i,x in enumerate(a) if x == 1]
        neg = [i for i,x in enumerate(a) if x == -1]
        for i in pos:
            for j in neg:
                cand = a[:]
                cand[i] = -cand[i]
                cand[j] = -cand[j]
                if abs(sum(cand)) == k:
                    return cand
        # if that fails, just fall back to a simple valid CW seed (rare)
        return make_simple_cw(n, k)

    return a


def make_simple_cw(n: int, k: int) -> List[int]:
    """
    Simple valid CW(n,k^2) seed:
      - exactly k^2 nonzeros
      - sum = ±k
      (no structure, for fallback use)
    """
    weight = k*k
    if weight > n:
        raise ValueError("k^2 > n")

    a = [0]*n
    support = random.sample(range(n), weight)
    target_sum = k if random.random() < 0.5 else -k
    if (weight + target_sum) % 2 != 0:
        target_sum = -target_sum
    p = (weight + target_sum)//2
    q = weight - p

    plus_positions = set(random.sample(support, p))
    for idx in support:
        a[idx] = 1 if idx in plus_positions else -1
    return a


def support_key(a):
    """Key for smart pruning & tabu: positions of nonzeros."""
    return tuple(i for i, x in enumerate(a) if x != 0)


def random_1swap(a):
    """Swap one nonzero with one zero (preserves weight & sum)."""
    n = len(a)
    a = a[:]
    sup = [i for i, x in enumerate(a) if x != 0]
    zer = [i for i, x in enumerate(a) if x == 0]
    if not sup or not zer:
        return a
    i = random.choice(sup)
    j = random.choice(zer)
    a[i], a[j] = a[j], a[i]
    return a


def random_2swap(a):
    """Swap two nonzeros with two zeros (preserves weight & sum)."""
    n = len(a)
    a = a[:]
    sup = [i for i, x in enumerate(a) if x != 0]
    zer = [i for i, x in enumerate(a) if x == 0]
    if len(sup) < 2 or len(zer) < 2:
        return a
    i1, i2 = random.sample(sup, 2)
    j1, j2 = random.sample(zer, 2)
    a[i1], a[j1] = a[j1], a[i1]
    a[i2], a[j2] = a[j2], a[i2]
    return a

def best_1swap_local(a, max_candidates=None):
    """
    Try 1-swaps and return best neighbor (or (None, inf) if none better).
    max_candidates: limit number of pairs for larger n; if None → exhaustive.
    """
    sup = [i for i, x in enumerate(a) if x != 0]
    zer = [i for i, x in enumerate(a) if x == 0]

    pairs = list(itertools.product(sup, zer))
    if max_candidates is not None and len(pairs) > max_candidates:
        pairs = random.sample(pairs, max_candidates)

    baseE = corr_energy_numba(a)
    best = None
    bestE = baseE

    for i, j in pairs:
        cand = a[:]
        cand[i], cand[j] = cand[j], cand[i]
        E = corr_energy_numba(cand)
        if E < bestE:
            bestE = E
            best = cand

    if best is None:
        return None, baseE
    return best, bestE


def best_2swap_local(a, max_candidates=None):
    """
    Try 2-swaps (two support ↔ two zero) and return best neighbor.
    """
    sup = [i for i, x in enumerate(a) if x != 0]
    zer = [i for i, x in enumerate(a) if x == 0]

    if len(sup) < 2 or len(zer) < 2:
        return None, corr_energy_numba(a)

    sup_pairs = list(itertools.combinations(sup, 2))
    zer_pairs = list(itertools.combinations(zer, 2))

    pairs = list(itertools.product(sup_pairs, zer_pairs))
    if max_candidates is not None and len(pairs) > max_candidates:
        pairs = random.sample(pairs, max_candidates)

    baseE = corr_energy_numba(a)
    best = None
    bestE = baseE

    for (i1, i2), (j1, j2) in pairs:
        cand = a[:]
        cand[i1], cand[j1] = cand[j1], cand[i1]
        cand[i2], cand[j2] = cand[j2], cand[i2]
        E = corr_energy_numba(cand)
        if E < bestE:
            bestE = E
            best = cand

    if best is None:
        return None, baseE
    return best, bestE



def path_relink_preserving(A, B, max_steps=200):
    """
    Path relinking from A toward B using only swaps:
      - swaps 0 <-> nonzero where A differs from B
      - swaps +1 <-> -1 where signs differ
    Always preserves weight & sum.

    Returns best solution encountered along the path.
    """
    A = A[:]
    n = len(A)
    best = A[:]
    bestE = corr_energy_numba(A)

    for _ in range(max_steps):
        # classify mismatches
        idx0 = []    # A=0,   B!=0
        idx1 = []    # A!=0,  B=0
        pos_mis = [] # A=+1,  B=-1
        neg_mis = [] # A=-1,  B=+1

        for i in range(n):
            if A[i] == B[i]:
                continue
            if A[i] == 0 and B[i] != 0:
                idx0.append(i)
            elif A[i] != 0 and B[i] == 0:
                idx1.append(i)
            elif A[i] == 1 and B[i] == -1:
                pos_mis.append(i)
            elif A[i] == -1 and B[i] == 1:
                neg_mis.append(i)

        moves = []

        # 0 <-> nonzero swaps
        for i in idx0:
            for j in idx1:
                moves.append((i, j))

        # sign swaps
        for i in pos_mis:
            for j in neg_mis:
                moves.append((i, j))

        if not moves:
            break

        best_step = None
        bestE_step = float("inf")

        for i, j in moves:
            cand = A[:]
            cand[i], cand[j] = cand[j], cand[i]
            E = corr_energy_numba(cand)
            if E < bestE_step:
                bestE_step = E
                best_step = cand

        if best_step is None or bestE_step >= bestE:
            # no improving step along path
            break

        A = best_step
        best = A[:]
        bestE = bestE_step

    return best, bestE




def tabu_search_cw(
    a0,
    max_iters=20000000,
    tabu_tenure=20,
    no_improve_limit=1000,
    elite_size=5,
    ls_1swap_limit=None,   # e.g. None for n=24, or 5000 for n=48
    ls_2swap_limit=None,
    seed_fn=None           # function () -> new valid seed, for restarts
):
    """
    Tabu + strong local search + path relinking for CW(n,k^2).

    a0: initial valid sequence (list[int] with values in {-1,0,1})
    """

    n = len(a0)
    current = a0[:]
    currentE = corr_energy_numba(current)

    best = current[:]
    bestE = currentE

    tabu = {}  # support_key -> expire_iter
    visited_supports = {}  # support_key -> best_energy_seen
    elite = [best[:]]      # archive of best solutions

    stagnation = 0
    it = 0

    while it < max_iters:
        it += 1
        print("energy: " , currentE)

        # record support & smart pruning info for current
        sk_cur = support_key(current)
        prev = visited_supports.get(sk_cur, float("inf"))
        if currentE < prev:
            visited_supports[sk_cur] = currentE

        # update global best and elite
        if currentE < bestE:
            bestE = currentE
            best = current[:]
            elite.append(best[:])
            elite = sorted(elite, key=corr_energy_numba)[:elite_size]
            stagnation = 0
        else:
            stagnation += 1

        # early exit if perfect
        if bestE == 0:
            break

        # --------- Strong local search (VND style) ----------
        improved = False

        # Only do full local search if n is small or occasionally for larger n
        if n <= 32 or it % 50 == 0 or currentE <= 100:
            # best 1-swap
            ls1, ls1E = best_1swap_local(current, ls_1swap_limit)
            if ls1 is not None and ls1E < currentE:
                current, currentE = ls1, ls1E
                improved = True

            if not improved:
                # best 2-swap
                ls2, ls2E = best_2swap_local(current, ls_2swap_limit)
                if ls2 is not None and ls2E < currentE:
                    current, currentE = ls2, ls2E
                    improved = True

        if improved:
            continue  # go to next iteration with improved current

        # --------- Build neighbors (tabu-guided) ----------
        neighbors = []

        # a few random 1-swaps
        for _ in range(10):
            neighbors.append((random_1swap(current), "1swap"))

        # a few random 2-swaps
        for _ in range(5):
            neighbors.append((random_2swap(current), "2swap"))

        best_neighbor = None
        best_neighbor_E = float("inf")

        for cand, mtype in neighbors:
            sk = support_key(cand)

            # tabu check
            if sk in tabu and tabu[sk] > it:
                continue

            E = corr_energy_numba(cand)

            # smart pruning: if we've seen this support with <= energy, skip
            prevE = visited_supports.get(sk, float("inf"))
            if prevE <= E:
                continue

            if E < best_neighbor_E:
                best_neighbor_E = E
                best_neighbor = cand

        # if no admissible neighbor, try an escape move or restart
        if best_neighbor is None:
            # simple escape: random 3-swap
            esc = random_2swap(random_1swap(current))
            escE = corr_energy_numba(esc)
            current, currentE = esc, escE
            stagnation += 1

            # optional: restart if stuck too long
            if seed_fn is not None and stagnation > no_improve_limit:
                current = make_grouped_seed(n,6)
                currentE = corr_energy_numba(current)
                stagnation = 0
                tabu.clear()
                visited_supports.clear()
                elite = [current[:]]
            continue

        # move to best neighbor
        current = best_neighbor
        currentE = best_neighbor_E

        # update tabu
        sk_new = support_key(current)
        tabu[sk_new] = it + tabu_tenure
        # purge expired
        tabu = {k: v for k, v in tabu.items() if v > it}

        # store best energy for this support
        prevE = visited_supports.get(sk_new, float("inf"))
        if currentE < prevE:
            visited_supports[sk_new] = currentE

        # ------------ Path relinking occasionally ------------
        if elite and it % 200 == 0:
            elite_partner = random.choice(elite)
            pr_sol, prE = path_relink_preserving(current, elite_partner)
            if prE < currentE:
                current, currentE = pr_sol, prE

    return {
        "best_seq": best,
        "best_energy": bestE,
        "iterations": it,
        "elite": elite,
    }

def solve(n:int,weight:int,maxrestarts = 100):
    k = int(math.sqrt(weight))
    print(f"\n{'='*40}")
    print(f"Solving CW ({n}, {weight}) -> n={n}, k={k}")
    print(f"\n{'-'*40}")
    
    maxruntime = 300 if n>= 48 else 60
    iterlimits = 1000
    maxiter = 5000000
    
    limit1swap = 5000 if n > 100 else None
    limit2swap = 10000 if n > 100 else None
    
    for attempt in range(1,maxrestarts+1):
        print(f"Attempt {attempt} (Time limit: {maxruntime}s)")
        
        arr = make_grouped_seed(n,k)
        starttime = time.time()
        
        result = tabu_search_cw(
        arr,
        max_iters=20000000,
        tabu_tenure=n,
        no_improve_limit=1000,
        elite_size=5,
        ls_1swap_limit=None if n < 24 else 5000,
        ls_2swap_limit=None if n < 100 else 100000,
        seed_fn=maxruntime 
        )
        
        runtime = time.time() - starttime
        
        energy = result["best_energy"]
        seq = result["best_seq"]
        
        print(f"\n > Best energy: {energy} (Time: {runtime:.2f}.s)")
        
        if energy == 0:
            print(f"\n Solution found in Attempt {attempt}")
            print(f"Sequence: {seq}")
            validation = validate_cw(seq,k)
            print(f"Validation: {validation.valid}")
            break
    return
    
def main():
    solve(24,9)
    solve(28,4)
    solve(48,36)
    solve(142,100)
if __name__ == "__main__":
    main()
