import { backtrackingSearch, annealingSearch, validateCW } from './cw';

function showSequence(label: string, seq: number[]) {
  console.log(label, seq.join(','));
}

function run() {
  const cases: [number, number][] = [ [24,3], [28,2], [48,6], [142,10] ];
  for (const [n,k] of cases) {
    console.log(`\n=== Case n=${n}, k=${k} (weight=${k*k}) ===`);
    const bt = backtrackingSearch(n, k, { timeLimitMs: 2000, maxSolutions: 1, preferPositive: true });
    if (bt.solutions.length) {
      console.log('Backtracking solution found. Explored states:', bt.explored);
      showSequence('BT Sequence:', bt.solutions[0]);
    } else {
      console.log('Backtracking failed or timed out; switching to annealing');
      const an = annealingSearch(n, k, { iterations: 50000 });
      console.log('Annealing energy:', an.energy, 'valid:', an.foundValid, 'iterations:', an.iterations);
      showSequence('Annealing candidate:', an.sequence);
    }
  }
  // verify provided examples
  const ex1 = [0,0,0,-1,-1,0,0,0,0,0,1,-1,0,0,0,-1,1,0,0,1,0,0,-1,-1];
  const ex2 = [-1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0];
  console.log('\nValidate example CW(24,9):', validateCW(ex1,3));
  console.log('Validate example CW(28,4):', validateCW(ex2,2));
}

run();
