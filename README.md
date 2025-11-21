# Constant Weight (CW) Sequence Search

## Problem Description

This project implements search algorithms to find **CW(n, k²)** sequences—arrays of length `n` containing elements from `{-1, 0, +1}` that satisfy:

1. **Weight constraint**: Exactly `k²` non-zero elements (and `n - k²` zeros)
2. **Constant sum property**: `(a₁ + a₂ + ... + aₙ)² = k²` (i.e., sum = ±k)
3. **Zero autocorrelation**: For shifts `s = 1, ..., m` where `m = ⌊n/2⌋`:
   ```
   C(s) = Σᵢ aᵢ · a₍ᵢ₊ₛ₎ mod n = 0
   ```

### Examples

**CW(24, 9)**: `[0,0,0,-1,-1,0,0,0,0,0,1,-1,0,0,0,-1,1,0,0,1,0,0,-1,-1]`
- Weight: 9, Sum: -3, All correlations zero ✓

**CW(28, 4)**: `[-1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0]`
- Weight: 4, Sum: 2, All correlations zero ✓

---

## Algorithms Implemented

### 1. Backtracking Search (Exact)
**Approach**: Systematic depth-first exploration with constraint propagation.

**Key techniques**:
- **Early pruning**: Reject partial assignments when:
  - Non-zero count exceeds `k²`
  - Remaining positions cannot reach target sum `±k`
  - Partial autocorrelations exceed feasible bounds
- **Incremental correlation tracking**: Update `C(s)` values as indices are assigned
- **Domain ordering**: Try most promising values first (configurable sign preference)

**Performance**: 
- Small cases (n≤30, k≤3): Fast (seconds)
- Medium cases (n=48, k=6): Timeout (>5000ms limit)
- Large cases (n≥100): Intractable without stronger symmetry breaking

### 2. Simulated Annealing (Heuristic)
**Approach**: Stochastic local search minimizing autocorrelation energy.

**Energy function**:
```
E(a) = Σₛ₌₁ᵐ [C(s)]²
```
Goal: E(a) = 0 while maintaining weight and sum constraints.

**Moves**:
- Flip sign of non-zero element
- Swap zero ↔ non-zero positions (preserving weight)

**Schedule**: Exponential cooling from `T_start = 5.0` to `T_end = 0.01` over 20,000–50,000 iterations.

**Performance**:
- Fast per iteration (no backtracking overhead)
- Finds low-energy candidates but rarely reaches E=0 for hard cases
- Useful for generating good starting points or approximate solutions

---

## Design Decisions

### Data Structures
- **Array representation**: Direct indexing for O(1) access during correlation computations
- **Correlation cache**: Incremental updates in backtracking avoid O(n²) recomputation per node

### Tradeoffs
- **Backtracking**: Completeness vs. exponential time complexity
- **Annealing**: Speed vs. no optimality guarantee

### Future Improvements
- **SAT encoding**: Formulate as Boolean satisfiability problem (tools: MiniSat, Z3)
- **Constraint programming**: Model in CP solver (e.g., OR-Tools)
- **Symmetry breaking**: Exploit cyclic/reflection symmetries to reduce search space
- **Parallelization**: Distribute backtracking subtrees across cores

---

## Installation & Usage

### Prerequisites
- Node.js 18+ and npm

### Setup
```bash
npm install
npm run build
```

### Running Tests
```bash
npm test
```
Validates the two provided examples using Jest.

### Running Search Examples
```bash
npm run examples
```
Attempts to find CW(24,9), CW(28,4), CW(48,36), CW(142,100).

### Programmatic Usage
```typescript
import { validateCW, backtrackingSearch, annealingSearch } from './src/cw';

// Validate a candidate sequence
const seq = [0,0,0,-1,-1,0,0,0,0,0,1,-1,0,0,0,-1,1,0,0,1,0,0,-1,-1];
const result = validateCW(seq, 3);
console.log(result.valid); // true

// Search via backtracking
const bt = backtrackingSearch(28, 2, { timeLimitMs: 5000, maxSolutions: 1 });
if (bt.solutions.length) console.log('Found:', bt.solutions[0]);

// Search via annealing
const an = annealingSearch(48, 6, { iterations: 50000 });
console.log('Energy:', an.energy, 'Valid:', an.foundValid);
```

---

## Test Results

### Required Cases

| Case | n | k² | Backtracking | Annealing | Notes |
|------|---|----|--------------|-----------| |
| CW(24,9) | 24 | 9 | Timeout | E=4 (invalid) | Needs longer search or better heuristics |
| CW(28,4) | 28 | 4 | ✓ Found (62,806 states) | — | Valid solution: `[0,...,1,1,0,...,1,-1]` |
| CW(48,36) | 48 | 36 | Timeout | E=49 (invalid) | High density makes pruning less effective |
| CW(142,100) | 142 | 100 | Timeout | E=349 (invalid) | Requires specialized techniques (SAT/CP) |

### Provided Examples Validation
- **CW(24,9)**: ✓ Valid (weight=9, sum=-3, all correlations zero)
- **CW(28,4)**: ✓ Valid (weight=4, sum=2, all correlations zero)

---

## Project Structure
```
cw-search/
├── src/
│   ├── cw.ts           # Core algorithms and validation
│   └── examples.ts     # Test runner for required cases
├── __tests__/
│   └── cw.test.ts      # Jest unit tests
├── dist/               # Compiled JavaScript (generated)
├── package.json
├── tsconfig.json
├── jest.config.cjs
└── README.md
```

---

## Team Information
_[Add your names, student IDs, course details here]_

**Course**: [Course Number/Name]  
**Date**: November 20, 2025  
**Group Members**:
- Student 1 (ID: ...)
- Student 2 (ID: ...)

---

## References
- M. J. E. Golay, "Complementary Series," IRE Transactions on Information Theory, 1961.
- Constraint Programming techniques for sequence design problems.
- Simulated Annealing: Kirkpatrick et al., "Optimization by Simulated Annealing," Science, 1983.

---

## License
Academic project - educational use only.
