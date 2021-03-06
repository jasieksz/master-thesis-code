#%%
from typing import List, NamedTuple, Tuple, Dict
import numpy as np
from itertools import combinations,chain
from time import time
import copy
import pandas as pd

from definitions import Profile, Candidate, Voter
from vcrDetectionAlt import findCRPoints, detectVCRProperty, createGPEnv, findVRPoints
from ch7_main import deleteVoters, deleteCandidates, getVCLists
from static_profiles import VCRNCOP_55_1, VCRNCOP_55_2, VCRNCOP_55_3, VCRNCOP_66, VCRNCOP_1010,VCR_1515_01k, VCR_77_0, VR_77_0, VCR_1212_0
from static_profiles import VCR_CV_S
from mavUtils import getVCROrders,getVCRProfileInCRVROrder,getVCRProfileInCROrder,getVCRProfileInVROrder,getVCREndOrders
from vis_vcr import vcrProfileToAgentsWithDeletion, plotVCRAgents, vcrProfileToAgentsWithColors, vcrProfileToAgents
from vrUtils import getFullProfileVRFromVCR
from vcrDomain import isVCR


#%%
#
# Helper methods
#
class avScore(NamedTuple):
    cId:int
    score:int

def flatten(arr):
    return list(chain(*arr))
    cc
def voteCount(A:np.ndarray, c:int) -> int:
    return int(sum(A[:,c]))

def avElection(A:np.ndarray) -> List[Tuple[int,int]]:
    idScores = sorted(zip(range(A.shape[1]), np.sum(A, axis=0)), key=lambda i_s: -i_s[1])
    return list(map(lambda i_s: avScore(i_s[0], i_s[1]), idScores))
    
def scoreDelta(A:np.ndarray, c1:int, c2:int) -> int:
    return voteCount(A, c1) - voteCount(A, c2)

columnOnesIndex = lambda arr,column: np.where(arr[:,column] == 1)[0] # List[int], indices of 1s in a given column
rowOnesIndex = lambda arr,row: np.where(arr[row] == 1)[0] # List[int], indices of 1s in a given row
descSecondSort = lambda tuple: -tuple[1] # Sorting key for Tuple2, descending
tupleIn = lambda tup,key: key in tup # check if key is equal to any element of the tuple
tupleContains = lambda tup,keys: [tupleIn(tup,key) for key in keys] # check if keys from the collection are in the given tuple

#
# Bruteforce AV-CC-DV - returns all possible combinations of voters to delete (of minimum size)
#
def cc_dv_brute(A:np.ndarray, p:int) -> List[List[int]]:
    
    deletedVoters = []
    k = 1 # combination counter
    N,M = A.shape # voters, candidates
    whiteList = columnOnesIndex(A, p)

    avScores = avElection(A)
    if avScores[0][0] == p and avScores[0][1] != avScores[1][1]:
        return True, deletedVoters

    while(k < N):
        voterCombination = [comb for comb in combinations(range(N), k) if not any(tupleContains(comb, whiteList))]
        for comb in voterCombination:
            tmpA = np.array(A) # make a copy
            tmpA[list(comb)] = 0 # delete votes
            avScores = avElection(tmpA)
            if avScores[0][0] == p and avScores[0][1] != avScores[1][1]: # if p wins election tmpA
                deletedVoters.append(list(comb))
        if len(deletedVoters) > 0:
            return True, deletedVoters
        k += 1
    return len(deletedVoters) != 0, deletedVoters

#
# ConstructiveControl-DeleteVoters - naive greedy approach
#
def cc_dv_naive_broken(A:np.ndarray, p:int, deletedVoters:List[int]) -> List[int]:
    
    # voters, candidates
    N,M = A.shape 
    
    # voters we cannot delete
    votersWhiteList = columnOnesIndex(arr=A, column=p) 

    # AV score for candidate p
    pScore = sum(A[:,p]) 

    # sorted (score desc.) candidates with score >= scoreP and not p
    # avElection(A) returns List[Tuple(candidate, score)], sorted descending by score
    opponents = [candScore[0] for candScore in avElection(A) if candScore[1] >= pScore]
    opponents = [opponent for opponent in opponents if opponent != p]
        
    # if p has no opponents then p is the winner
    if len(opponents) == 0:
        return True,deletedVoters

    # opponent with highest score
    nemesis = opponents[0]

    # voters who approve of nemesis and are not in the whitelist
    nemesisVoters = [v for v in columnOnesIndex(arr=A, column=nemesis) if v not in votersWhiteList]

    # how many voters we need to delete for p to beat nemesis
    votersToDeleteCount = scoreDelta(A, c1=nemesis, c2=p) + 1

    # if there is not enough voters to delete then p cannot win
    if len(nemesisVoters) < votersToDeleteCount:
        return False,deletedVoters

    # opponents approved by nemesisVoters
    nemesisVoterApprovedOpponents= [(nV,[c for c in rowOnesIndex(arr=A, row=nV) if c in opponents]) for nV in nemesisVoters]
    nemesisVoterApprovedOpponentsCount = [(nV, len(op)) for nV,op in nemesisVoterApprovedOpponents]
    
    # nemesisVoters sorted desc. by approved opponents count
    nemesisVotersOpponentCount = sorted(nemesisVoterApprovedOpponentsCount, key=descSecondSort)
    
    # delete voters (set to vector 0)
    votersToDelete = [nv[0] for nv in nemesisVotersOpponentCount[:votersToDeleteCount]]
    A[votersToDelete] = 0
    deletedVoters.append(votersToDelete)

    # repeat
    return cc_dv_naive_broken(A, p, deletedVoters)

#
# fixed naive greedy solution - recalculating scores after each deletion
#
def cc_dv_naive_fixed(A:np.ndarray, p:int, deletedVoters:List[int]) -> List[int]:
    N,M = A.shape # voters, candidates
    
    # voters we cannot delete - O(n)
    votersWhiteList = columnOnesIndex(arr=A, column=p)

    # AV score for candidate p - O(n)
    pScore = sum(A[:,p])

    # sorted (score desc.) candidates with score >= scoreP and not p
    # avElection(A) returns List[Tuple(candidate, score)], sorted descending by score
    # O(nmlogm) - tak wystarczy max() zamist sort() (w avElection) czyli O(nm) 
    opponents = [candScore[0] for candScore in avElection(A) if candScore[1] >= pScore]
    opponents = [opponent for opponent in opponents if opponent != p]
        
    # if p has no opponents then p is the winner
    if len(opponents) == 0:
        return True,deletedVoters

    # opponent with highest score
    nemesis = opponents[0]

    # voters who approve of nemesis and are not in the whitelist - O(n)
    nemesisVoters = [v for v in columnOnesIndex(A, nemesis) if v not in votersWhiteList]

    # how many voters we need to delete for p to beat nemesis
    votersToDeleteCount = scoreDelta(A, nemesis, p) + 1

    # if there is not enough voters to delete then p cannot win
    if len(nemesisVoters) < votersToDeleteCount:
        return False,deletedVoters

    # opponents approved by nemesisVoters - O(n * (nm + n + nm + n)) -> O(n^2 * m)
    tmpDelete = []
    for vD in range(votersToDeleteCount):
        # we need to recalculate avElection after every deletion, bcs.
        # nemesisVoterApprovedOpponents order might have changed (some c might not be opponent anymore after deleting v_{i-1})
        opponents = [candScore[0] for candScore in avElection(A) if candScore[1] >= pScore] # O(nmlogm), wystarczy max() -> O(nm)
        opponents = [opponent for opponent in opponents if opponent != p]

        nemesisVoters = [v for v in columnOnesIndex(A, nemesis) if v not in votersWhiteList] # O(n)
        nemesisVoterApprovedOpponents = [(nV,[c for c in rowOnesIndex(A, nV) if c in opponents]) for nV in nemesisVoters] # O(nm)
        nemesisVoterApprovedOpponentsCount = [(nV, len(op)) for nV,op in nemesisVoterApprovedOpponents] # O(n)
        
        # nemesisVoters sorted desc. by approved opponents count - O(nlogn), wystarczy max() -> O(n)
        nemesisVotersOpponentCount = sorted(nemesisVoterApprovedOpponentsCount, key=descSecondSort)
        
        # delete voter (set row to 0s) # O(m)
        A[nemesisVotersOpponentCount[0][0]] = 0
        tmpDelete.append(nemesisVotersOpponentCount[0][0])
    
    deletedVoters.append(tmpDelete)
    # repeat
    return cc_dv_naive_fixed(A, p, deletedVoters) # n * O(n^2 * m) - O(n^3 * m)

#
# Utility to compare results from different algos
#
def resultComparator(combsNaive, combsBrute):
    if len(combsNaive) == 0 and len(combsBrute) == 0:
        return True
    combsNaive = sorted(flatten(combsNaive))
    return combsNaive in [sorted(comb) for comb in combsBrute]

#
# Verify algorithms vs bruteforce algo is an instance of greedy cc-dv
#
def compareAlgos(profiles, vCount:int, algo):
    result = []
    failedMsg = "p={}, i={}\n{}\nresAlgo={}\nresBrute={}\n"
    for p in range(vCount):
        for i,profile in enumerate(profiles):
            status1, combs1 = algo(copy.deepcopy(profile), p, [])
            status2, combs2 = cc_dv_brute(np.array(profile.A), p)

            if (status1 != status2) or ((status1 and status2) and not resultComparator(combs1, combs2)):
                # print(failedMsg.format(p,i,profile.A,combs1,combs2))
                result.append((p,i,combs1,combs2))
                pass
    return result


#%%
#
# ConstructiveControl-DeleteVoters - vcr greedy sweeping
#
def getOpponents(A, p):
    pScore = sum(A[:,p]) 
    opponents = [candScore[0] for candScore in avElection(A) if candScore[1] >= pScore]
    return [opponent for opponent in opponents if opponent != p]
    
def sortOpponentsByVCRBegining(profile:Profile, opponents:List[int]) -> List[int]:
    _,candOrder = getVCROrders(profile)
    return [cand for cand in candOrder if cand in opponents]

def sortVotersByVCREnd(profile:Profile, voters:List[int]) -> List[int]:
    voteOrder,_ = getVCREndOrders(profile)
    return [voter for voter in voteOrder if voter in voters]

#%%
def cc_dv_vcr(P:Profile, p:int, deletedVoters:List[int]) -> List[int]:
    
    # voters, candidates
    N,M = P.A.shape 
    # voters we cannot delete
    votersWhiteList = columnOnesIndex(arr=P.A, column=p) 
    
    opponents = getOpponents(A=P.A, p=p)
    if len(opponents) == 0:
        return True,deletedVoters

    opponents = sortOpponentsByVCRBegining(P, opponents)
    nemesis = opponents[0]
    nemesisVoters = [v for v in columnOnesIndex(arr=P.A, column=nemesis) if v not in votersWhiteList]
    nemesisVoters = sortVotersByVCREnd(profile=P, voters=nemesisVoters)
    nemesisVoters.reverse()

    # how many voters we need to delete for p to beat nemesis
    delta = scoreDelta(P.A, c1=nemesis, c2=p) + 1
    # if there is not enough voters to delete then p cannot win
    if len(nemesisVoters) < delta:
        return False,deletedVoters

    # delete voters (set to vector 0)
    P.A[nemesisVoters[:delta]] = 0
    deletedVoters.append(nemesisVoters[:delta])

    # repeat
    return cc_dv_vcr(P, p, deletedVoters)

#%%
#
# Helpers
#

def colorGenerator(cStr:str, vStr:str, profile:Profile, p:int):
    opponents = getOpponents(A=profile.A, p=p)
    pVoters = columnOnesIndex(arr=profile.A, column=p)
    opponentsVoters = set(flatten([columnOnesIndex(profile.A,c) for c in opponents]))

    opC = {cStr + str(op):'red' for op in opponents}
    vWhiteList = {vStr + str(pV):'lightskyblue' for pV in pVoters}
    vBlackList = {vStr + str(oV):'blue' for oV in opponentsVoters if not oV in pVoters}

    return {cStr + str(p):'gold', **opC, **vWhiteList, **vBlackList}

def viableControlElections(profiles:List[Profile]) -> Dict[int,List[int]]:
    pRange = profiles[0].A.shape[1]
    return {
        pId:list(
                map(lambda t2: t2[0],
                filter(lambda t3: t3[1][0] and len(t3[1][1]) > 0,
                    ((i,cc_dv_brute(np.copy(p.A),pId)) for i,p in enumerate(profiles))
                ))
            )
        for pId in range(pRange)
    }

def getCCProfile(gEnv) -> Profile:
    ccA = np.array([0,0,0,0,0,1,
                    0,0,1,0,0,1,
                    0,1,1,0,0,1,
                    0,1,1,0,0,0,
                    1,1,1,0,0,0,
                    1,1,1,0,1,0,
                    1,0,0,1,1,0,
                    0,0,0,1,1,0,
                    0,0,0,1,0,0]).reshape(9,6)

    Vs,Cs = getVCLists(ccA)
    ilpStatus, ilpRes = detectVCRProperty(ccA, Cs, Vs, gEnv)
    return Profile.fromILPRes(ccA, ilpRes, Cs, Vs)

#%%

#
# Notebook
#

#%%
gEnv = createGPEnv()
# ccP = getCCProfile(gEnv)

# P66 = VCRNCOP_66()
# P66_CC = viableControlElections(P66)

#%%
VR77 = VR_77_0()[:100]
VCR1212 = VCR_1212_0()

#%%
VR77Ordered = [p for s,p in (getFullProfileVRFromVCR(p, gEnv) for p in VR77) if s]
VR77_CC = viableControlElections(VR77Ordered)

#%%
VCR1212_CC = viableControlElections(VCR1212[:100])

#%%
VCR1212_CC[0]

#%%
start = time()
print(compareAlgos(VR77Ordered, 7, cc_dv_vcr))
print(time() - start)

#%%
P = VR77Ordered[:100]

#%%
i

#%%
j = 1
p = 0
i = VCR1212_CC[p][j]
P = VCR1212

print(P[i].A)
print("\n", sum(P[i].A), "\n")
print("BRUTE : ", cc_dv_brute(np.copy(P[i].A), p=p))
print("VCR : ", cc_dv_vcr(copy.deepcopy(P[i]), p=p, deletedVoters=[]))

plotVCRAgents(vcrProfileToAgentsWithColors(P[i], colorGenerator('C', 'V', P[i], p)))

#%%
# def run(start:int, end:int):
#     P1515 = VCR_1515_01k()
#     print("Start {} End {}".format(s,e))
#     startTime = time()
#     res = compareAlgos(P1515[start:end], 15, cc_dv_vcr)
#     print("Total time {}".format(time() - startTime))
#     print("Reuslt\n", res)

#     import sys

#     if __name__ == "__main__":
#         s = int(sys.argv[1])
#         e = int(sys.argv[2])
#         run(s,e)  


#%%
p1 = VR77[1]
_,p5 = getFullProfileVRFromVCR(p1, gEnv)

#%%
Vs, Cs = getVCLists(p5.A)
s,d = findVRPoints(p5.A, Cs, Vs, gEnv)
d

#%%
p6 = Profile.fromILPRes(shuffleCols(p5.A, [0,1,2,3,4,5,6]), d, Cs, Vs)
print(p6)

#%%
i = 1
p = 6
tmpP = VCR1212[i]
print(tmpP.A)
print("BRUTE : ", cc_dv_brute(np.copy(tmpP.A), p=p))
print("VCR : ", cc_dv_vcr(copy.deepcopy(tmpP), p=p, deletedVoters=[]))
plotVCRAgents(vcrProfileToAgentsWithColors(tmpP, colorGenerator('C', 'V', tmpP, p)))


#%%
def shuffleCols(array:np.ndarray, order:list) -> np.ndarray:
    A = np.array(array).transpose()
    A[list(range(A.shape[0]))] = A[order]
    return A.transpose()

#%%
isVCR(p5)

#%%
def crPlot():
    A = np.array([1,1,1,1,0,
                  1,0,1,0,0,
                  1,1,0,1,0,
                  0,0,0,0,1]).reshape(4, 5)
    C = [Candidate("c0", 2, 2),
         Candidate("c1", 1, 0.5),
         Candidate("c2", 4, 1),
         Candidate("c3", 2, 1.25),
         Candidate("cP", 6, 0.5)]
    V = [Voter("v0", 2, 1),
         Voter("v1", 4, 0.5),
         Voter("v2", 1.5, 0.2),
         Voter("v3", 6, 0.2)]
    return Profile(A, C, V)

colors = {'cP':'gold',
        'c0':'red', 'c1':'red', 'c2':'red', 'c3':'red',
        'v0':'blue', 'v1':'blue', 'v2':'blue',
        'v3':'lightblue'}

plotVCRAgents(vcrProfileToAgentsWithColors(crPlot(), colors))


#%%
def dist(profiles):
    cx = np.empty(0, dtype=float)
    cr = np.empty(0, dtype=float)
    vx = np.empty(0, dtype=float)
    vr = np.empty(0, dtype=float)

    for profile in profiles:
        for cand in profile.C:
            cx = np.append(cx, cand.x)
            cr = np.append(cr, cand.r)
        for voter in profile.V:
            vx = np.append(vx, voter.x)
            vr = np.append(vr, voter.r)
    return cx,cr,vx,vr

#%%
sums = lambda profiles: [np.sum(p.A) / (p.A.shape[0] * p.A.shape[1]) for p in profiles]


#%%
cx,cr,vx,vr = dist(VCR1212)

#%%
s2020 = sums(VCR_CV_S(c=20, v=20, s=0))
s1010 = sums(VCR_CV_S(c=10, v=10, s=0))

#%%
df1010 = pd.DataFrame(zip(s1010, np.ones(len(s1010))), columns=['sum', 'type'])
df2020 = pd.DataFrame(zip(s2020, np.zeros(len(s1010))), columns=['sum', 'type'])
df = pd.concat([df1010, df2020])

#%%
sns.displot(data=df, x='sum', hue='type')

#%%
