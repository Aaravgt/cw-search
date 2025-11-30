import time
from main import solver

def run_tests():
    tests = [
        (24,9),
        (28,4),
        (48,36),
        (142,100),
    ]
    
    for i, (n,ksquared) in enumerate(tests, 1):
        print(f"Test case{i}: n={n}, k={ksquared}")
        
        print("-"*40)
        starttime = time.time()
        result = solver(n,ksquared,1000000,600)
        totaltime = time.time()-starttime
        
        print(f"result: {result}")
        print(f"Time: {totaltime:.2f}s")
        print()

run_tests()