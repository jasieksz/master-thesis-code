#%%
import numpy as np
from definitions import Profile, Candidate, Voter, VCR44
from itertools import combinations
from typing import List
import math
from numpy import ndarray

#%% PROFILES
def CR_44(): # <=> VI <=> foreach v in V : v_r = 0
     return np.array([0,0,1,0,1,0,1,0,1,1,0,1,1,0,0,1]).reshape(4,4)

#%% HELPERS
def candsToLetters(C):
    return [chr(ord('A') + c) for c in C]

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

def mavScore(A: np.ndarray, committee: np.array) -> int:
    partialHD = lambda preference: hammingDistance(committee, preference)
    return max([partialHD(pref) for pref in A])

def minimax(A: np.ndarray, k: int) -> np.array:
    candidateCount = A.shape[1]
    scores = [(mavScore(A, committeeTupleToVector(committee, candidateCount)), [chr(ord('A') + i) for i in committee])
                for committee in combinations(range(candidateCount), k)]
    scores.sort(key=lambda mavScore_committee: mavScore_committee[0])
    return scores

def minimaxCR(A, k, d):
    n,m = A.shape # Vn Cm
    C = [i for i in range(m)]
    K = []
    for voterIndex in range(n):
        if len(K) == k:
            return K

        X = [candidate for candidate in C if candidate not in K]

        vectorK = committeeTupleToVector(committee=K, C=m)
        s = hammingDistance2(vectorK, A[voterIndex]) + k - len(K)
        if s > d:
            r = math.ceil((s - d) / 2)
            if r > k - len(K) or r > len(X):
                print("NOT FOUND")
                return []
            
            X_sorted = getCandsSortedByVoteFromVoter(A, X, voterIndex)
            for c in X_sorted[:r]:
                print("Adding {} to {}".format(candsToLetters([c])[0], K))
                K.append(c)

    if k > len(K):
        print("ARBITRARY SELECTION - Current K = {}".format(K))
        for c in [c for c in C if c not in K][:(k-len(K))]:
            K.append(c)
    return K

#%%
A = CR_44()
print(A)

#%%
minimax(A, 2)

#%%
mav = minimaxCR(A, k=2, d=2)
mav.sort()
candsToLetters(mav)