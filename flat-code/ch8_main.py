#%%
from typing import List, NamedTuple, Tuple
import numpy as np
from itertools import combinations,chain
from time import time

from definitions import Profile, Candidate, Voter
from vcrDetectionAlt import findCRPoints, detectVCRProperty, createGPEnv
from ch7_main import deleteVoters, deleteCandidates, getVCLists
from ch7_main import VCRNCOP_55_1, VCRNCOP_55_2, VCRNCOP_55_3, VCRNCOP_66, VCRNCOP_1010
from mavUtils import getVCROrders


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
    failedMsg = "p={}, i={}\n{}\nres1={}\nres2={}\n"
    for p in range(vCount):
        for i,profile in enumerate(profiles):
            status1, combs1 = algo(np.array(profile.A), p, [])
            status2, combs2 = cc_dv_brute(np.array(profile.A), p)

            if (status1 != status2) or ((status1 and status2) and not resultComparator(combs1, combs2)):
                print(failedMsg.format(p,i,profile.A,combs1,combs2))
                pass

# start = time()
# compareAlgos(VCRNCOP_66(), 6, cc_dv_naive_fix)
# print(time() - start)

#%%
#
# ConstructiveControl-DeleteVoters - vcr greedy sweeping
#
def getOpponents(A, p):
    pScore = sum(A[:,p]) 
    opponents = [candScore[0] for candScore in avElection(A) if candScore[1] >= pScore]
    return [opponent for opponent in opponents if opponent != p]
    
def sortOpponentsByVCR(profile:Profile, opponents:List[int]) -> List[int]:
    _,candOrder = getVCROrders(profile)
    return [cand for cand in candOrder if cand in opponents]

def sortVotersByVCR(profile:Profile, voters:List[int]) -> List[int]:
    voteOrder,_ = getVCROrders(profile)
    return [voter for voter in voteOrder if voter in voters]

def cc_dv_vcr(P:Profile, p:int, deletedVoters:List[int]) -> List[int]:
    
    # voters, candidates
    N,M = P.A.shape 
    # voters we cannot delete
    votersWhiteList = columnOnesIndex(arr=P.A, column=p) 
    
    opponents = getOpponents(A=P.A, p=p)
    if len(opponents) == 0:
        return True,deletedVoters

    opponents = sortOpponentsByVCR(P, opponents)
    nemesis = opponents[0]
    nemesisVoters = [v for v in columnOnesIndex(arr=P.A, column=nemesis) if v not in votersWhiteList]
    nemesisVoters = sortVotersByVCR(profile=P, voters=nemesisVoters)
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


#
# Notebook
#

#%%
gEnv = createGPEnv()

#%%
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
ccP = Profile.fromILPRes(ccA, ilpRes, Cs, Vs)

#%%

start = time()
compareAlgos(VCRNCOP_66(), 6, cc_dv_naive_broken)
print(time() - start)

#%%
a = np.array(P66[0].A)
print(a)
b = deleteVoters(a, [1,2])
print(a)
b

#%%
a = [0,1,2,3,4,5]

#%%
cc_dv_vcr(P=ccP,p=0,deletedVoters=[])


#%%
ccP