import math
import random
import time


def isvalidparam(n,ksquared):
    #check if problem is even valid
    if ksquared > n:
        return False
    
    return True
    
    
def isvalidCW(seq,ksquared):#check if it meets all cw requirements
    n = len(seq)
    k = math.isqrt(ksquared)
    #1 check weight
    nonzerosum = sum(1 for i in seq if i!=0)
    if nonzerosum!= ksquared:
        return False
    
    #2 check sum
    total=sum(seq)
    if total*total != ksquared:
        return False
    
    #3 check autocorrelations
    m = n // 2 if n % 2 == 0 else (n-1) // 2
    for num in range(1, m+1):
        corr = 0
        for i in range(n):
            j = (i + num) % n
            corr += seq[i] * seq[j]
        if corr!=0:
            return False
    return True

def backtracking(n,ksquared,maxtime ):#fins valid sequence
    k = math.isqrt(ksquared)
    
    if not isvalidparam(n,k):
        return "no solution found given invalid parameters"
    
    targetweight = ksquared
    sequence = [0]*n #initialize sequence wirh zero
    startime = time.time()
    
    for targetsum in [k, -k]:#try positive and negative values
        result = recursive_backtracking(sequence, 0,0,0,n,k,targetweight,targetsum,startime,maxtime)
        if result is not None and result!= "No solution found":
            return result
        if time.time()-startime > maxtime:
            break
    return "No solution found"

def recursive_backtracking(sequence,pos,used,currentsum,n,k,targetweight,targetsum,startime,maxtime):# recursive herlper
    #time check
    if time.time()-startime > maxtime:
        return None
    
    #check if valid after completing ssequence
    if pos == n:
        if used == targetweight and currentsum == targetsum:
            if isvalidCW(sequence,targetweight):
                return sequence
        return None
    
    remaining = n - pos
    needed = targetweight - used
    
    #prune if we cant reach target weight(not enough positions left)
    if needed > remaining:
        return None
    
    #prune if we cant reach target sum(summ too large)
    if abs(currentsum) > k:
        return None
    
    #try all possible values
    for value in [0,1,-1]:
        #skip if we cant add anymore non-zero's
        if value!=0 and used >= targetweight:
            continue
        
        #update sequence
        sequence[pos] = value
        newused = used + (1 if value != 0 else 0)
        newsum = currentsum + value
        
        #recursively explore next iotions
        result = recursive_backtracking(sequence, pos +1,newused,newsum,n,k,targetweight,targetsum,startime,maxtime)
        
        if result!=None:
            return result
        
        #backtrack
        sequence[pos]=0
    return None

def createinitialsolution(n,weighttarget,targetsum):
    sequence = [0]*n
    
    #try to create valid initial solution
    for attempt in range(10): #try 10 times
        positions = random.sample(range(n),weighttarget)
        pluscount = (weighttarget + targetsum) // 2
        minuscount = (weighttarget - targetsum) // 2
        
        #check if counts are valid
        if pluscount * 2== (weighttarget + targetsum) and minuscount * 2==(weighttarget - targetsum):
            if 0<= pluscount <= weighttarget and 0 <= minuscount <=weighttarget:
                pluspos = random.sample(positions, pluscount)
                minuspos = [p for p in positions if p not in pluspos]
                
                #assign +1 and -1 values
                for pos in pluspos:
                    sequence[pos] = 1
                for pos in minuspos:
                    sequence[pos]= -1
                return sequence
        targetsum = -targetsum # try negative if positivr fordnt work
        
    #fallback return random sequence if precise solution fails
    sequence = [0]*n
    positions = random.sample(range(n),weighttarget)
    for pos in positions:
        sequence[pos] = random.choice([-1,1])
    return sequence

def objectivefunction(sequence,ksquared):
    n = len(sequence)
    k = math.isqrt(ksquared)
    
    #calculate sum of aquare correlations(ideal = 0 for perfect solution)
    totalcorreror = 0
    m = n//2 if n%2 == 0 else (n-1)//2
    for shift in range(1,m+1):
        correlation = 0
        for i in range(n):
            j = (i+shift)%n
            correlation+= sequence[i]*sequence[j]
        totalcorreror+= correlation **2
    
    #penalty for weight mismatch
    currentweight = sum(1 for x in sequence if x!=0)
    weightpenalty = (currentweight - ksquared)**2 * 1000
    
    #penalty for sum mismatch
    currentsum = sum(sequence)
    sumpenalty = (currentsum**2 - ksquared)**2 * 1000
    return totalcorreror+weightpenalty+sumpenalty

def generateneighbour(current):
    n = len(current)
    neighbour = current.copy()
    
    #identify different positions
    nonzeropos = [i for i,x in enumerate(neighbour) if x!=0]
    zeropos = [i for i,x in enumerate(neighbour) if x==0]
    possiblemoves = []
    
    #identify positions of different types
    if nonzeropos and zeropos:
        possiblemoves.append("swap-nz-z")
    if nonzeropos:
        possiblemoves.append("flipsign")
    if len(nonzeropos)>=2:
        possiblemoves.append("swap-nz-nz")
    if not possiblemoves:
        return neighbour,[],[],[]
    
    #determine possible moves 
    move = random.choice(possiblemoves)
    if move == "swap-nz-z":
        i = random.choice(nonzeropos)
        j = random.choice(zeropos)
        pos = [i,j]
        oldval = [neighbour[i],neighbour[j]]#cacth old and new values
        newval = [neighbour[j],neighbour[i]]
        neighbour[i],neighbour[j] = neighbour[j], neighbour[i] # swap
        return neighbour,pos,oldval,newval
        
    elif move == "flipsign":
        pos = random.choice(nonzeropos)
        oldval = neighbour[pos]
        newval = -oldval
        neighbour[pos] = newval
        return neighbour,[pos],[oldval],[newval]
    
    elif move=="swap-nz-nz":
        i,j = random.sample(nonzeropos, 2)
        pos = [i,j]
        oldval = [neighbour[i],neighbour[j]]#cacth old and new values
        newval = [neighbour[j],neighbour[i]]
        neighbour[i],neighbour[j] = neighbour[j], neighbour[i]
        return neighbour,pos,oldval,newval 
        
    return neighbour,[],[],[]

def calculatedeltaenergy(n,currentsequence,currentcorelations,poslist,oldval,newval):
    #helps approximate enrgy change
    m = n//2 if n%2 == 0 else (n-1)//2
    deltacorrenergy = 0
    newcorrelations = list(currentcorelations)
    
    #update correlations
    for idx,p in enumerate(poslist):
        oldv = oldval[idx]
        newv = newval[idx]
        diff = newv - oldv
        
        if diff == 0: continue
        
        for shift in range(1,m+1):
            jfor = (p+shift)%n
            jback = (p - shift)%n
            
            termfwd = currentsequence[jfor]
            termback = currentsequence[jback]
            
            change = diff * (termfwd + termback)
            
            #update the ctored vale for correlation
            oldcorr = newcorrelations[shift-1]
            newcorr = oldcorr + change
            newcorrelations[shift-1]=newcorr
            
            #add change
            deltacorrenergy+=(newcorr**2 - oldcorr**2)
    return deltacorrenergy,newcorrelations
        
    
    
def annealing(n,ksquared,iterations= 10000000):
    k = math.isqrt(ksquared)
    feasible = isvalidparam(n,k)
    if not feasible:
        return "No solution found given invalid parameters"
    
    weighttarget = ksquared
    targetsum = k
    
    #try multiple inital solution
    bestoverall = None
    bestenergy = float('inf')
    
    for attempt in range(10):# try 3 diff sol
        #initalize current solution
        current = createinitialsolution(n,weighttarget, targetsum)
        currentenergy = objectivefunction(current,ksquared)
    
        m = n // 2 if n%2 == 0 else (n-1)//2
        currentcorrelations = []
        for shift in range(1,m+1):
            corr = 0
            for i in range(n):
                j = (i+shift)%n
                corr+= current[i]*current[j]
            currentcorrelations.append(corr)
            
        best = current.copy()
        bestenergysofar = currentenergy
    
        # annealing parameters
        initialtemp = 1000.0
        alpha = 0.999
        mintemp = 0.001
        currenttemp = initialtemp
        
        #main loop
        for iteration in range(iterations):
            #generate neighbour
            neighbour,movedpos,oldval,newval = generateneighbour(current)
            
            if not movedpos: continue 
            deltaecorr,newcorr = calculatedeltaenergy(n,current,currentcorrelations,movedpos,oldval,newval)
            
            currentpen = (sum(1 for x in current if x!=0) - ksquared)**2*100000
            neighbourpen = (sum(1 for x in neighbour if x!=0) - ksquared)**2*100000
            
            currentspen = (sum(current)**2-ksquared)**2 * 100000
            neighbourspen = (sum(neighbour)**2-ksquared)**2 * 100000
            
            deltapen = (neighbourpen+neighbourspen)-(currentpen+currentspen)
            deltaenergy = deltaecorr + deltapen
            neighbourenergy = deltaenergy+currentenergy
            
            #acceptance criterion
            if deltaenergy < 0: # accept improving moves
                current = neighbour
                currentenergy = neighbourenergy
                currentcorrelations = newcorr #update correlations
                
                if neighbourenergy < bestenergysofar:# best so far == local maxima: 
                    best = neighbour.copy()
                    bestenergysofar = neighbourenergy
            elif currenttemp > mintemp:# accept worse moves with probability to escape local maxima
                acceptance = math.exp(-deltaenergy / currenttemp)
                if random.random() < acceptance:
                    current = neighbour
                    currentenergy = neighbourenergy
                    currentcorrelations = newcorr
                
            # cooling mechanism,also we'll drift to prevent too much drift
            currenttemp = max(mintemp,alpha * currenttemp)  
            if iteration % 1000 == 0:
                currentenergy = objectivefunction(current, ksquared)
                currentcorrelations = []
                for shift in range(1,m+1):
                    corr = 0
                    for i in range(n):
                        j = (i+shift)%n
                        corr+= current[i]*current[j]
                    currentcorrelations.append(corr)
            
            if bestenergysofar == 0 and isvalidCW(best, ksquared):#check if best sol found
                return best
            
        if bestenergysofar < bestenergy:
            bestenergy = bestenergysofar
            bestoverall = best
    return bestoverall if bestoverall is not None and isvalidCW(bestoverall,ksquared) else "no solution found"

def solver(n,ksquared,annealingiterations,timeout):
    k = math.isqrt(ksquared)
    print(f"solving CW{n}, {ksquared} with k={k}")
    #validate param first
    if not isvalidparam(n,ksquared):
        return "No solution found with invalid parameters"
    
    #try  annealing first
    print("Annealing first")
    starttime = time.time()
    result = annealing(n,ksquared,annealingiterations)
    annealingtime = time.time() - starttime
    
    if isinstance(result, list):
        print(f"annealing solution found in {annealingtime:.2f}s")
        return result
    
    print("annealing failed or parameters invalid")
    print("Try backtracking")
    
    starttime = time.time()
    result = backtracking(n,ksquared,timeout)
    backtrackingtime = time.time() - starttime
    if isinstance(result, list):
        print(f"Backtracking solution found in {backtrackingtime:.2f}")  
        return result
    else:
        print("backtracking failed")  
    return result
    