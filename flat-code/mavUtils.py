#%%
import numpy as np
from definitions import Profile, Candidate, Voter, VCR44
from itertools import combinations
from typing import List, Tuple, Dict
import math
from numpy import ndarray
from vcrDetectionAlt import findCRPoints, detectVCRProperty
from collections import namedtuple
from crUtils import getProfileCRFromVCR
from pprint import pprint
from utils import shuffleRows, shuffleCols
from typing import NamedTuple, Any

#%% Static status
STATUS_FAIL = -1
STATUS_OK = 1
STATUS_APPROX = 2
STATUS_OTHER = 3
NO_SCORE = -7

class MAVCmpResult(NamedTuple):
    status:int
    W1:List[List[int]]
    W2:List[List[int]]
    score1:int
    score2:int
    other:Any

#%% HELPERS
def getCandsSortedByVotes(A, X):
    b = [(i, sum(A[:,i])) for i in range(A.shape[1])]
    b.sort(key=lambda t: -t[1])
    return [t[0] for t in b if t[0] in X]

def getCandsAfter(C:list, after:int):
    return [c for c in C if c > after]

def committeeTupleToVector(committee:tuple, C:int) -> np.ndarray:
    committeeArray = np.zeros(C)
    np.add.at(committeeArray, list(committee), 1)
    return committeeArray
    
def getCandsSortedByVoteFromVoter(A:ndarray, X:list, v:int) -> list:
    candidateVoteCount = [(c, sum(A[v:,c])) for c in range(A.shape[1])]
    candidateVoteCount.sort(key=lambda t: -t[1])
    return [t[0] for t in candidateVoteCount if t[0] in X]

#%% MAV
def hammingDistance(a: np.ndarray, b: np.ndarray) -> int:
    return (a != b).nonzero()[0].shape[0]
    
def hammingDistance2(a: np.array, b: np.array) -> int:
    return sum(np.logical_xor(a, b))

def mavScore(A: np.ndarray, committee: np.array) -> int: # O(nm)
    partialHD = lambda preference: hammingDistance(committee, preference)
    return max([partialHD(vote) for vote in A])

#
# BRUTE FORCE MAV
# 
def minimaxFull(A: np.ndarray, k: int) -> Tuple[int, List[List[int]]]:
    candidateCount = A.shape[1]
    scores = [(mavScore(A, committeeTupleToVector(committee, candidateCount)), committee)
                for committee in combinations(range(candidateCount), k)]
    scores.sort(key=lambda mavScore_committee: mavScore_committee[0])
    return STATUS_OK, scores                

def minimax(A: np.ndarray, k: int) -> Tuple[int, List[List[int]]]:
    candidateCount = A.shape[1]
    scores = [(mavScore(A, committeeTupleToVector(committee, candidateCount)), committee)
                for committee in combinations(range(candidateCount), k)]
    scores.sort(key=lambda mavScore_committee: mavScore_committee[0])
    return STATUS_OK, list(filter(lambda ms_c: ms_c[0] == scores[0][0], scores)),
                    
#
# Liu& Guo CR MAV
# 
def minimaxCR(A, k, d) -> Tuple[int, List[int]]:
    n,m = A.shape # Vn Cm
    C = [i for i in range(m)]
    K = []
    for voterIndex in range(n):
        if len(K) == k:
            return STATUS_OK, K

        X = [candidate for candidate in C if candidate not in K and A[voterIndex,candidate] == 1]
        vectorK = committeeTupleToVector(committee=K, C=m)
        s = hammingDistance2(vectorK, A[voterIndex]) + k - len(K)
        if s > d:
            r = math.ceil((s - d) / 2)
            if r > k - len(K):
                return STATUS_FAIL,[]
            if r > len(X):
                return STATUS_FAIL,[]
            
            X_sorted = getCandsSortedByVoteFromVoter(A, X, voterIndex+1)
            for c in X_sorted[:r]:
                K.append(c)

    if k > len(K):
        for c in [c for c in C if c not in K][:(k-len(K))]:
            K.append(c)
        return STATUS_APPROX, K
    return STATUS_OK, K

def saveArray(path, array):
    with open(path, 'wb') as f:
        np.save(file=f, arr=array, allow_pickle=False)

cmprMAVRes = namedtuple('cmprMAVRes',['status', 'score', 'kX', 'kY'])

def basePartialCompare(A:np.ndarray, algoX, algoY) -> Tuple[bool, int, list, list]:
    statusX, wX = algoX(A)
    statusY, wY = algoY(A)
    sX = mavScore(A, committeeTupleToVector(wX[0][1], A.shape[1])) if statusX != STATUS_FAIL else NO_SCORE
    sY = mavScore(A, committeeTupleToVector(wY, A.shape[1])) if statusY != STATUS_FAIL else NO_SCORE

    if sX == sY and sY != NO_SCORE:
        return MAVCmpResult(status=statusY,
                            W1=wX,
                            W2=wY,
                            score1=sX,
                            score2=sY,
                            other=None)
    elif sX != sY:
        return MAVCmpResult(status=STATUS_FAIL,
                            W1=wX,
                            W2=wY,
                            score1=sX,
                            score2=sY,
                            other=None)

def analyze(profile, k, d):
    print(profile, "\n")
    crSol = minimaxCR(profile, k, d)
    bruteSol = minimax(profile, k)
    print("Brute")
    pprint(bruteSol)
    print("\nCR")
    pprint(crSol)

def getVCROrders(profile):                                                                                                                                                                                                                                                            
    V, C = list(profile.V), list(profile.C)
    V.sort(key=lambda t3: t3[1] - t3[2]) # x - r
    C.sort(key=lambda t3: t3[1] - t3[2]) # x - r
    V = list(map(lambda voter: int(voter.id[1:]), V))
    C = list(map(lambda voter: int(voter.id[1:]), C))
    return V, C

def getVCREndOrders(profile):                                                                                                                                                                                                                                                            
    V, C = list(profile.V), list(profile.C)
    V.sort(key=lambda t3: t3[1] + t3[2]) # x + r
    C.sort(key=lambda t3: t3[1] + t3[2]) # x + r
    V = list(map(lambda voter: int(voter.id[1:]), V))
    C = list(map(lambda voter: int(voter.id[1:]), C))
    return V, C

def shuffleVC(A:np.ndarray, voterOrder:List[int], candOrder:List[int]) -> np.ndarray:
    return shuffleCols(shuffleRows(A, voterOrder), candOrder)

def getVCRProfileInCROrder(profile:Profile) -> Profile:
    oV, oC = getVCROrders(profile)
    oA = shuffleRows(profile.A, oV)
    return Profile(A=oA, C=profile.C, V=profile.V)

def getVCRProfileInVROrder(profile:Profile) -> Profile:
    oV, oC = getVCROrders(profile)
    oA = shuffleCols(profile.A, oC)
    return Profile(A=oA, C=profile.C, V=profile.V)

def getVCRProfileInCRVROrder(profile:Profile) -> Profile:
    oV, oC = getVCROrders(profile)
    oA = shuffleVC(A=profile.A, voterOrder=oV, candOrder=oC)
    return Profile(A=oA, C=profile.C, V=profile.V)