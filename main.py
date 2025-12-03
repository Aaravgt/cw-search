import random, math
from typing import List, Tuple, Dict, Any
import numpy as np
from numba import njit
from dataclasses import dataclass
from typing import List, Optional, Tuple
import itertools
from numpy.fft import fft, ifft 

#Douglas-Raachfford Projection Operators
def P_A_ternary(x: np.ndarray) -> np.ndarray:
    """Projection onto the set A (ternary values egg{-1, 0, 1})."""
    # Simple rounding
    return np.clip(np.round(x), -1, 1).astype(np.float64)


def P_B_weight_sum(x: np.ndarray, k_sq: int, k: int) -> np.ndarray:
    """
    Projection onto the set B (Weight k^2 and Sum +/-k).
    Selects k^2 largest magnitude elements and adjusts signs for the sum constraint.
    """
    n = len(x)
    k_sq = int(k_sq)
    k = int(k)
    
    #Selection (k^2 non-zero elements)
    abs_x = np.abs(x)
    sorted_indices = np.argsort(abs_x)[::-1]
    support_indices = sorted_indices[:k_sq]
    
    y = np.zeros(n, dtype=np.float64)
    
    #Selected elements are set to their original sign
    y[support_indices] = np.sign(x[support_indices])
    y[support_indices][y[support_indices] == 0] = 1.0 

    # Siign Adjustment 
    current_sum = np.sum(y)
    
    # Calculate target sum based on sequence's
    target_sum = k if current_sum >= 0 else -k
    diff = int(current_sum - target_sum)

    # Number of sign flips required
    num_flips = abs(diff) // 2

    if diff > 0: # Current sum is too high flip +1 into -1
        plus_indices = np.where(y == 1.0)[0]
        if len(plus_indices) >= num_flips:
            to_flip = np.random.choice(plus_indices, num_flips, replace=False)
            y[to_flip] = -1.0
    elif diff < 0: # Current sum is too low flip -1 into +1
        minus_indices = np.where(y == -1.0)[0]
        if len(minus_indices) >= num_flips:
            to_flip = np.random.choice(minus_indices, num_flips, replace=False)
            y[to_flip] = 1.0
            
    return y.astype(np.float64)


def P_C_autocorrelation(x: np.ndarray, k_sq: int) -> np.ndarray:
    """
    Projection onto the set C (Zero Autocorrelation, |DFT(x)|^2 = k^2).
    """
    n = len(x)
    target_mag = math.sqrt(float(k_sq))
    
    x_hat = fft(x)
    
    # Enforce Spectral Constraint
    abs_x_hat = np.abs(x_hat)
    
    # Calculate scale factor
    scale_factor = np.full(n, target_mag) / (abs_x_hat + 1e-10) 
    
    # Apply scaling to the  coefficients
    y_hat = x_hat * scale_factor

    # We take the real part as the original sequence is real
    y = ifft(y_hat).real
    
    return y.astype(np.float64)


def douglas_rachford_intensification(x: List[int], k_sq: int, k: int, num_steps: int = 100) -> List[int]:
    """
    Hybrid Feasibility step: Iteratively projects the sequence closer to the
    feasible region (A intersection B intersection C).
    """
    x_float = np.array(x, dtype=np.float64)
    
    for _ in range(num_steps):
        # 1. Project onto Ternary (PA)
        x_proj = P_A_ternary(x_float)

        # 2. Project onto Weight/Sum (PB)
        x_proj = P_B_weight_sum(x_proj, k_sq, k)
        
        # 3. Project onto Zero Autocorrelation (PC)
        x_proj = P_C_autocorrelation(x_proj, k_sq)
        
        # Simpler Projection for non-convex sets.
        x_float = x_proj
        
    # The final output must be rounded to the ternary alphabet for the Tabu Search
    return P_A_ternary(x_float).astype(np.int64).tolist()


# Numba accelerated func (Optimization)


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

#Validation and seeding 

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

    # Check periodic autocorrelations
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
        # useful when abs sum not checked
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

    # distribute weight evenly
    Wc = [weight//d]*d
    for i in range(weight % d):
        Wc[i] += 1

    a = [0]*n
    target_sum = k if random.random() < 0.5 else -k
    # Ensure It's even
    if (weight + target_sum) % 2 != 0:
        target_sum = -target_sum 
        
    p = (weight + target_sum)//2
    
    # distribution of pluses
    p_left = p
    q_left = weight - p

    for c in range(d):
        w = Wc[c]
        if w==0: continue

        chosen = random.sample(cosets[c], w)
        
        # Proportional calculation
        p_c_target = int(round(w * p / weight)) if weight > 0 else 0
        p_c = min(p_c_target, p_left, w)

        p_left -= p_c

        pluses = set(random.sample(chosen, p_c))

        for idx in chosen:
            a[idx] = 1 if idx in pluses else -1

    return a


def make_structured_cw_coset(n: int, k: int) -> List[int]:
    """
    Structured CW(n, k^2) seed with:
      exactly k^2 nonzeros
      sum = ±k
      Support distributed across residue classes mod d (cosets)
    """
    weight = k * k
    if weight > n:
        return "IS INVALID"

    nice_divs = [16, 12, 8, 6, 4, 3, 2]
    divs = [d for d in nice_divs if n % d == 0]
    d = divs[0] if divs else 1 

    cosets = [[] for _ in range(d)]
    for i in range(n):
        cosets[i % d].append(i)

    base = weight // d
    rem  = weight % d
    coset_weights = [base] * d
    for c in range(rem):
        coset_weights[c] += 1

    target_sum = k if random.random() < 0.5 else -k
    if (weight + target_sum) % 2 != 0:
        target_sum = -target_sum
    p_total = (weight + target_sum) // 2
    
    # distribute plus counts
    plus_per_coset = [0] * d
    if weight > 0:
        frac = [cw * p_total / weight for cw in coset_weights]
        plus_per_coset = [int(round(x)) for x in frac]
        
        diff = sum(plus_per_coset) - p_total
        if diff != 0:
            indices = list(range(d))
            random.shuffle(indices)
            step = 1 if diff > 0 else -1
            diff = abs(diff)
            for idx in indices:
                if diff == 0:
                    break
                new_val = plus_per_coset[idx] - step
                if 0 <= new_val <= coset_weights[idx]:
                    plus_per_coset[idx] = new_val
                    diff -= 1

    a = [0] * n
    p_left = p_total

    for c in range(d):
        w_c = coset_weights[c]
        if w_c == 0:
            continue
        positions = cosets[c]

        support_c = random.sample(positions, w_c)
        p_c = min(plus_per_coset[c], p_left, w_c)
        p_left -= p_c

        plus_positions = set(random.sample(support_c, p_c))
        for idx in support_c:
            a[idx] = 1 if idx in plus_positions else -1
            
    # uses simple seed if structural fails)
    s = sum(a)
    w = sum(1 for x in a if x != 0)
    if abs(s) != k or w != weight:
        return make_simple_cw(n, k)

    return a


def make_simple_cw(n: int, k: int) -> List[int]:
    """Simple valid CW(n,k^2) seed ."""
    weight = k*k
    if weight > n:
        raise ValueError("k^2 > n")

    a = [0]*n
    support = random.sample(range(n), weight)
    target_sum = k if random.random() < 0.5 else -k
    if (weight + target_sum) % 2 != 0:
        target_sum = -target_sum
    p = (weight + target_sum)//2
    
    plus_positions = set(random.sample(support, p))
    for idx in support:
        a[idx] = 1 if idx in plus_positions else -1
    return a


def support_key(a):
    """Key for smart pruning & tabu: positions of nonzeros."""
    return tuple(i for i, x in enumerate(a) if x != 0)


#  Neighbourhood search algo's ()

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
    Path relinking from A toward B using only swaps that preserve weight & sum.
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

        # 0 and nonzero swaps (maintain weight)
        for i in idx0:
            for j in idx1:
                moves.append((i, j))

        # sign swaps (maintain sum)
        for i in pos_mis:
            for j in neg_mis:
                moves.append((i, j))

        if not moves:
            break
            
        # choose best improving move
        best_step = None
        bestE_step = float("inf")

        # Limit move search for large problems
        if len(moves) > 1000:
            moves = random.sample(moves, 1000)

        for i, j in moves:
            cand = A[:]
            cand[i], cand[j] = cand[j], cand[i]
            E = corr_energy_numba(cand)
            if E < bestE_step:
                bestE_step = E
                best_step = cand

        if best_step is None:
            break

        A = best_step
        if bestE_step < bestE:
            best = A[:]
            bestE = bestE_step

    return best, bestE


# main tabu search func (I Modified for the douglas Hybrid)

def tabu_search_cw(
    a0,
    max_iters=20000000,
    tabu_tenure=20,
    no_improve_limit=1000,
    elite_size=5,
    ls_1swap_limit=None,
    ls_2swap_limit=None,
    dr_freq=500,           # Frequency of Douglas-Rachford Intensification
    dr_steps=100,          # Number of AP steps in DR
    seed_fn=None           # function () -> new valid seed, for restarts
):
    """
    Hybrid Tabu Search with Strong Local Search, Path Relinking, and
    Douglas-Rachford Intensification.
    """

    n = len(a0)
    current = a0[:]
    currentE = corr_energy_numba(current)
    
    # Calculate k and k_sq (needed for DR projections)
    k_sq = sum(1 for x in a0 if x != 0)
    k = math.isqrt(k_sq)

    best = current[:]
    bestE = currentE

    tabu = {}  # support_key -> expire_iter
    visited_supports = {}
    elite = [best[:]]

    stagnation = 0
    it = 0

    while it < max_iters:
        it += 1
        # Periodically log current status
        if it % 10000 == 0 or it == 1:
            print(f"Iter {it} | Best E: {bestE} | Current E: {currentE} | Stagnation: {stagnation}")

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
            print(f"\n SOLUTION FOUND at Iter {it}!")
            break

        # Douglas-Rachford Intensification (Hybrid) 
        if it % dr_freq == 0 and bestE > 0:
            # Run DR (Alternating Projections) on the current sequence
            dr_sol = douglas_rachford_intensification(current, k_sq, k, dr_steps)
            drE = corr_energy_numba(dr_sol)
            
            if drE < currentE:
                print(f"  > DR Intensification improved E from {currentE} to {drE} at Iter {it}")
                current, currentE = dr_sol, drE
                continue # Skip standard move selection for this iteration

        #Strong local search (VND style)
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

        # Build neighbors (tabu-guided) 
        neighbors = []
        for _ in range(10):
            neighbors.append(random_1swap(current))
        for _ in range(5):
            neighbors.append(random_2swap(current))

        best_neighbor = None
        best_neighbor_E = float("inf")

        for cand in neighbors:
            sk = support_key(cand)

            # tabu check (Aspiration: always accept if E < bestE)
            is_tabu = (sk in tabu and tabu[sk] > it)
            if is_tabu and corr_energy_numba(cand) >= bestE:
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
        if best_neighbor is None or stagnation > no_improve_limit:
            # simple escape: random 3-swap
            esc = random_2swap(random_1swap(current))
            escE = corr_energy_numba(esc)
            current, currentE = esc, escE
            stagnation += 1

            # optional: restart if stuck too long
            if seed_fn is not None and stagnation > no_improve_limit:
                # Use the calculated k for the restart seed
                current = seed_fn(n, k) 
                currentE = corr_energy_numba(current)
                print(f"\n STAGNATION. Restarting search (E={currentE}, Best E={bestE})...")
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
        tabu = {k_sk: v for k_sk, v in tabu.items() if v > it} # purge expired

        # store best energy for this support
        prevE = visited_supports.get(sk_new, float("inf"))
        if currentE < prevE:
            visited_supports[sk_new] = currentE

        # Path relinking occasionally 
        if elite and it % 200 == 0:
            elite_partner = random.choice(elite)
            pr_sol, prE = path_relink_preserving(current, elite_partner)
            if prE < currentE:
                print(f"  > Path Relinking improved E from {currentE} to {prE}")
                current, currentE = pr_sol, prE
                
                # If new global best, update elite set
                if prE < bestE:
                    best = pr_sol[:]
                    bestE = prE
                    elite.append(best[:])
                    elite = sorted(elite, key=corr_energy_numba)[:elite_size]

    return {
        "best_seq": best,
        "best_energy": bestE,
        "iterations": it,
        "elite": elite,
    }


def main(n,k):
    k=int(math.sqrt(k))
    arr = make_structured_cw_coset(n, k)
    print(f"{'-'*60}")
    print(f"Starting Search for CW({n}, {k*k}) with initial energy: {corr_energy_numba(arr)}")
    print(f"{'-'*60}")
    
    best = tabu_search_cw(
        arr, 
        max_iters=500000, 
        ls_1swap_limit=5000,
        ls_2swap_limit=10000,
        no_improve_limit=5000,
        dr_freq=1000,      # Run dr algo every 1000 iterations
        dr_steps=200,      # Take 200 steps in each dr intensification
        seed_fn=make_structured_cw_coset
    )
    
    val = validate_cw(best["best_seq"], k)
    
    print("\n--- Final Results ---")
    print(f"Problem: CW({n}, {k*k})")
    print(f"Best Sequence E: {best['best_energy']}")
    print(f"Validation: {'SUCCESS' if val.valid else 'FAILURE - ' + val.first_violation}")
    print(f"Solution: {best["best_seq"]}")
    print(f"Total Iterations: {best['iterations']}")

if __name__ == "__main__":
    main(24,9)
    main(28,4)
    main(48,36)
    main(148,100)
    
