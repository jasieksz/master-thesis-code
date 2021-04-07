#%%
from typing import List, NamedTuple, Tuple
import numpy as np
from itertools import combinations

from definitions import Profile, Candidate, Voter
from vcrDetectionAlt import findCRPoints, detectVCRProperty
from ch7_main import deleteVoters, deleteCandidates, getVCLists

#%%
def VCRNCOP_55() -> List[Profile]:
    A = np.load("resources/output/5C5V/NCOP-profiles/ncop-55-1.npy")
    return list(map(Profile.fromNumpy, A))


#%%
Ps = VCRNCOP_55()

#%%
class avScore(NamedTuple):
    cId:int
    score:int

def voteCount(A:np.ndarray, c:int) -> int:
    return int(sum(A[:,c]))

def avElection(A:np.ndarray) -> List[Tuple[int,int]]:
    idScores = sorted(zip(range(A.shape[1]), np.sum(A, axis=0)), key=lambda i_s: -i_s[1])
    return list(map(lambda i_s: avScore(i_s[0], i_s[1]), idScores))
    
#%%
# ConstructiveControl-DeleteVoters
def scoreDelta(A:np.ndarray, c1:int, c2:int) -> int:
    return voteCount(A, c1) - voteCount(A, c2)

def cc_dv_naive(A:np.ndarray, p:int, deletedVoters:List[int]) -> List[int]:

    #
    # Helper functions
    #
    columnOnesIndex = lambda arr,column: np.where(arr[:,column] == 1)[0]
    rowOnesIndex = lambda arr,row: np.where(arr[row] == 1)[0]
    descSecondSort = lambda tuple: -tuple[1]

    # voters, candidates
    N,M = A.shape 
    
    # voters we cannot delete
    votersWhiteList = columnOnesIndex(A,p)

    # AV score for candidate p
    pScore = sum(A[:,p])

    # sorted (score desc.) candidates with score >= scoreP and not p
    opponents = [candScore[0] for candScore in avElection(A) if candScore[1] >= pScore]
    opponents = [opponent for opponent in opponents if opponent != p]
        
    # if p has no opponents then p is the winner
    if len(opponents) == 0:
        return True,deletedVoters

    # opponent with highest score
    nemesis = opponents[0]

    # voters who approve of nemesis and are not in the whitelist
    nemesisVoters = [v for v in np.where(A[:,nemesis] == 1)[0] if v not in votersWhiteList]

    # how many voters we need to delete for p to beat nemesis
    votersToDeleteCount = scoreDelta(A, nemesis, p) + 1

    # if there is not enough voters to delete then p cannot win
    if len(nemesisVoters) < votersToDeleteCount:
        return False,deletedVoters

    # opponents approved by nemesisVoters
    nemesisVoterApprovedOpponents= [(nV,[c for c in rowOnesIndex(A,nV) if c in opponents]) for nV in nemesisVoters]
    nemesisVoterApprovedOpponentsCount = [(nV, len(op)) for nV,op in nemesisVoterApprovedOpponents]
    
    # nemesisVoters sorted desc. by approved opponents count
    nemesisVotersOpponentCount = sorted(nemesisVoterApprovedOpponentsCount, key=descSecondSort)
    
    # delete voters (set to vector 0)
    votersToDelete = [nv[0] for nv in nemesisVotersOpponentCount[:votersToDeleteCount]]
    A[votersToDelete] = 0
    deletedVoters.append(votersToDelete)

    # repeat
    return cc_dv_naive(A, p, deletedVoters)

#%%
i = 1
p = 1
a = np.array(Ps[i].A)
print(a)
cc_dv_naive(a, p, [])


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
ccA

#%%

#%%
len(Ps)

#%%
vs,cs = getVCLists(ccA)
vcrRes = detectVCRProperty(ccA, cs, vs)

#%%
ccP = Profile.fromILPRes(ccA, vcrRes[1], cs, vs)

#%%
print(ccP)

#%%
ccA_iter1 = deleteVoters(ccP.A, [2,1,3])
ccA_iter1

#%%
# list(zip(range(7),sum(ccA_iter1)))
list(zip(range(1,9),sum(ccA_iter1.transpose())))