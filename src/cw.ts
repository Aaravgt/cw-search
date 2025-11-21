/*
  Constant Weight (CW) sequence search utilities.
  A CW(n, k^2) sequence a[0..n-1] over {-1,0,+1} satisfies:
    1. Exactly k^2 non-zero entries.
    2. (sum_i a[i])^2 = k^2  => sum_i a[i] = +/- k.
    3. For shifts s = 1..m (m = floor(n/2)), periodic autocorrelation C(s) = sum_i a[i]*a[(i+s) mod n] = 0.

  We implement:
    - validateCW(sequence) -> boolean & details
    - backtrackingSearch(n, k) exact/partial search (k refers to sqrt of weight).
    - annealingSearch(n, k, options) heuristic local search.
*/

export interface CWValidationResult {
  valid: boolean;
  weight: number;
  targetWeight: number;
  sum: number;
  targetSumAbs: number;
  correlations: number[]; // C(1)..C(m)
  firstViolation?: string;
}

export function validateCW(a: number[], k: number): CWValidationResult {
  const n = a.length;
  const weight = a.filter(v => v !== 0).length;
  const targetWeight = k * k;
  const sum = a.reduce((acc, v) => acc + v, 0);
  const targetSumAbs = k;
  const m = Math.floor(n / 2);
  const correlations: number[] = [];
  let firstViolation: string | undefined;
  for (let s = 1; s <= m; s++) {
    let c = 0;
    for (let i = 0; i < n; i++) c += a[i] * a[(i + s) % n];
    correlations.push(c);
    if (firstViolation === undefined && c !== 0) firstViolation = `C(${s}) = ${c} != 0`;
  }
  if (firstViolation === undefined) {
    if (weight !== targetWeight) firstViolation = `weight ${weight} != ${targetWeight}`;
    else if (sum * sum !== targetWeight) firstViolation = `sum^2 ${sum * sum} != ${targetWeight}`;
    else if (Math.abs(sum) !== targetSumAbs) firstViolation = `|sum| ${Math.abs(sum)} != k ${k}`;
  }
  return { valid: firstViolation === undefined, weight, targetWeight, sum, targetSumAbs, correlations, firstViolation };
}

/* Backtracking search */
export interface BacktrackOptions {
  preferPositive?: boolean; // bias toward +1 first if true
  timeLimitMs?: number;
  maxSolutions?: number;
}

export interface BacktrackResult {
  solutions: number[][];
  explored: number;
  abortedByTime: boolean;
}

export function backtrackingSearch(n: number, k: number, opts: BacktrackOptions = {}): BacktrackResult {
  const targetWeight = k * k;
  const targetSumAbs = k;
  const m = Math.floor(n / 2);
  const correlations = new Array(m).fill(0);
  const start = Date.now();
  const timeLimit = opts.timeLimitMs ?? 5000;
  const maxSolutions = opts.maxSolutions ?? 1;

  const a = new Array(n).fill(0);
  let used = 0;
  let sum = 0;
  let explored = 0;
  const solutions: number[][] = [];

  function canStillReachSum(): boolean {
    const remaining = targetWeight - used; // max non-zero still placeable
    // sum must end at +/-k. If current sum = s, we can change by at most remaining in magnitude.
    return (Math.abs(targetSumAbs - sum) <= remaining) || (Math.abs(-targetSumAbs - sum) <= remaining);
  }

  function correlationsCanZero(position: number): boolean {
    // For each shift s, we have partial correlation value correlations[s-1]. Each new assignment of a[pos]
    // will affect future correlations when its pair enters. Precise pruning is complex; we use soft bound:
    // If current absolute correlation exceeds possible remaining contribution magnitude, prune.
    const remainingNonZeros = targetWeight - used;
    const maxAdditionalMagnitude = remainingNonZeros; // worst-case each additional pair contributes 1.
    for (let s = 1; s <= m; s++) {
      if (Math.abs(correlations[s - 1]) > maxAdditionalMagnitude) return false;
    }
    return true;
  }

  function placeValue(idx: number, v: number) {
    a[idx] = v;
    if (v !== 0) { used++; sum += v; }
    // update correlations incrementally where idx participates with earlier indices
    for (let s = 1; s <= m; s++) {
      const j = (idx - s + n) % n;
      if (j < idx) { // correlation pair becomes active
        correlations[s - 1] += a[idx] * a[j];
      }
    }
  }

  function removeValue(idx: number, v: number) {
    // reverse correlation contributions
    for (let s = 1; s <= m; s++) {
      const j = (idx - s + n) % n;
      if (j < idx) correlations[s - 1] -= a[idx] * a[j];
    }
    if (v !== 0) { used--; sum -= v; }
    a[idx] = 0;
  }

  const order = Array.from({ length: n }, (_, i) => i); // could reorder by symmetry

  function dfs(pos: number) {
    if (Date.now() - start > timeLimit) return true; // signal abort
    explored++;
    if (used > targetWeight) return false;
    if (!canStillReachSum()) return false;
    if (!correlationsCanZero(pos)) return false;
    if (pos === n) {
      if (used === targetWeight && (sum === targetSumAbs || sum === -targetSumAbs)) {
        // check correlations final
        for (let s = 1; s <= m; s++) if (correlations[s - 1] !== 0) return false;
        solutions.push([...a]);
        if (solutions.length >= maxSolutions) return true; // found enough
      }
      return false;
    }
    const idx = order[pos];
    // Remaining slots left
    const remainingSlots = n - pos;
    const needed = targetWeight - used;
    // If we need to fill all remaining with non-zero, enforce that.
    const mustBeNonZero = needed === remainingSlots;
    // Domain ordering
    const domain: number[] = [];
    if (!mustBeNonZero) domain.push(0);
    const signs = opts.preferPositive ? [1, -1] : [-1, 1];
    domain.push(...signs);
    for (const v of domain) {
      if (v !== 0 && used + 1 > targetWeight) continue;
      placeValue(idx, v);
      const abort = dfs(pos + 1);
      removeValue(idx, v);
      if (abort) return true;
    }
    return false;
  }
  const abortedByTime = dfs(0) && solutions.length === 0;
  return { solutions, explored, abortedByTime };
}

/* Simulated annealing / local search */
export interface AnnealOptions {
  iterations?: number;
  startTemp?: number;
  endTemp?: number;
  k?: number; // sqrt(weight)
  verboseEvery?: number;
  randomSeed?: number;
}

function makeRandomSequence(n: number, k: number, rng: () => number): number[] {
  const a = new Array(n).fill(0);
  const weight = k * k;
  // choose positions
  const positions = Array.from({ length: n }, (_, i) => i);
  for (let i = positions.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [positions[i], positions[j]] = [positions[j], positions[i]];
  }
  const chosen = positions.slice(0, weight);
  // assign signs to reach sum ±k (choose target sign randomly)
  const target = rng() < 0.5 ? k : -k;
  // start all -1 then flip some to +1 until sum matches target
  for (const p of chosen) a[p] = -1;
  let sum = -weight;
  const needDelta = target - sum; // we need to add needDelta total
  // flipping -1 to +1 adds +2 each
  const flipsNeeded = (needDelta + 2) / 2; // approximate; ensure integer
  let flips = 0;
  for (const p of chosen) {
    if (flips >= flipsNeeded) break;
    a[p] = 1;
    flips++;
  }
  return a;
}

function energy(a: number[]): number {
  const n = a.length;
  const m = Math.floor(n / 2);
  let e = 0;
  for (let s = 1; s <= m; s++) {
    let c = 0;
    for (let i = 0; i < n; i++) c += a[i] * a[(i + s) % n];
    e += c * c;
  }
  return e;
}

export interface AnnealResult {
  sequence: number[];
  energy: number;
  iterations: number;
  foundValid: boolean;
}

export function annealingSearch(n: number, k: number, options: AnnealOptions = {}): AnnealResult {
  const iterations = options.iterations ?? 20000;
  const startTemp = options.startTemp ?? 5.0;
  const endTemp = options.endTemp ?? 0.01;
  const seed = options.randomSeed ?? Date.now();
  let tRand = seed;
  const rng = () => {
    // simple LCG
    tRand = (tRand * 48271) % 0x7fffffff;
    return tRand / 0x7fffffff;
  };
  let a = makeRandomSequence(n, k, rng);
  let e = energy(a);
  for (let iter = 0; iter < iterations; iter++) {
    const temp = startTemp * Math.pow(endTemp / startTemp, iter / iterations);
    // propose: either swap sign of a non-zero or swap zero with non-zero preserving weight & sum constraints heuristically
    const indices = Array.from({ length: n }, (_, i) => i);
    const i = Math.floor(rng() * n);
    const j = Math.floor(rng() * n);
    const oldAi = a[i];
    const oldAj = a[j];
    // create candidate copy lazily
    let changed = false;
    if (oldAi !== 0 && oldAj !== 0) {
      // flip one sign
      a[i] = -a[i];
      changed = true;
    } else if (oldAi === 0 && oldAj !== 0) {
      // move non-zero
      a[i] = oldAj;
      a[j] = 0;
      changed = true;
    } else if (oldAi !== 0 && oldAj === 0) {
      a[j] = oldAi;
      a[i] = 0;
      changed = true;
    } else {
      // both zero -> set one to ±1 if still under weight limit (but weight would increase; not allowed). skip.
    }
    if (!changed) continue;
    // ensure weight & sum constraints preserved:
    const val = validateCW(a, k);
    if (!val.valid) {
      // revert if constraints grossly violated (weight or sum or correlations maybe). Use energy acceptance if only correlations off.
      if (val.weight !== k * k || Math.abs(val.sum) !== k) {
        // revert
        a[i] = oldAi;
        a[j] = oldAj;
        continue;
      }
    }
    const newE = energy(a);
    const delta = newE - e;
    if (delta <= 0 || rng() < Math.exp(-delta / Math.max(temp, 1e-6))) {
      e = newE;
      if (newE === 0 && val.valid) {
        return { sequence: [...a], energy: 0, iterations: iter + 1, foundValid: true };
      }
    } else {
      // revert
      a[i] = oldAi;
      a[j] = oldAj;
    }
  }
  const finalVal = validateCW(a, k);
  return { sequence: a, energy: e, iterations, foundValid: finalVal.valid };
}
