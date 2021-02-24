#%%
import numpy as np
from definitions import Profile, Candidate, Voter, VCR44
from itertools import combinations
from typing import List, Tuple, Dict
import math
from numpy import ndarray
from vcrDetectionAlt import findCRPoints
from collections import namedtuple
from crUtils import getProfileCRFromVCR
from pprint import pprint

#%% PROFILES
def CR_44(): # <=> VI <=> foreach v in V : v_r = 0
    p1 = np.array([0,0,1,0,1,0,1,0,1,1,0,1,1,0,0,1]).reshape(4,4)
    p2 = np.array([1,1,0,1,1,0,0,0,1,1,0,1,0,1,1,1]).reshape(4,4)
    p3 = np.array([1,0,1,0,1,1,1,1,0,1,1,1,0,0,1,1]).reshape(4,4)
    p4 = np.array([1,0,0,0,1,1,1,0,1,1,1,0,0,1,1,1]).reshape(4,4)
    return np.array([p1,p2,p3,p4])

     
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

def minimaxCR(A, k, d) -> Tuple[bool, List[int]]:
    n,m = A.shape # Vn Cm
    C = [i for i in range(m)]
    K = []
    for voterIndex in range(n):
        # print("LOOP VOTER : {}".format(voterIndex))
        if len(K) == k:
            return True, K

        X = [candidate for candidate in C if candidate not in K]

        vectorK = committeeTupleToVector(committee=K, C=m)
        s = hammingDistance2(vectorK, A[voterIndex]) + k - len(K)
        if s > d:
            r = math.ceil((s - d) / 2)
            if r > k - len(K):
                # print("NOT FOUND K")
                return False, []
            if r > len(X):
                # print("NOT FOUND X")
                return False,[]
            
            X_sorted = getCandsSortedByVoteFromVoter(A, X, voterIndex+1)
            for c in X_sorted[:r]:
                # print("Adding {} to {}".format(candsToLetters([c])[0], K))
                K.append(c)

    if k > len(K):
        # print("ARBITRARY SELECTION - Current K = {}".format(K))
        for c in [c for c in C if c not in K][:(k-len(K))]:
            K.append(c)
        return False,K
    return True, K

#%%
path = "resources/input/{}C{}V/VCR-{}.npy"

VCR_44_P = np.load(path.format(4,4,1))
VCR_44_A = np.array(list(map(lambda npp: Profile.fromNumpy(npp).A, VCR_44_P)))

#%%
CR_44_A = [pref for status,pref in (getProfileCRFromVCR(vcrp) for vcrp in VCR_44_A[:1000]) if status == True]

#%%
def compare(profiles: np.ndarray, k:int, crParamD:int):
    results = []
    for i, profile in enumerate(profiles):
        bruteMinScore = minimax(profile, k)[0][0]
        dDelta = 1
        status, K = minimaxCR(profile, k, crParamD)
        while not status and dDelta <= 4:
            print("++", i)
            status, K = minimaxCR(profile, k, crParamD + dDelta)
            dDelta +=1

        if status:
            crMinScore = mavScore(profile, committeeTupleToVector(K, profile.shape[1]))
            copyK = K
            while status and bruteMinScore != crMinScore and dDelta <= 4:
                print("**", i)
                status, K = minimaxCR(profile, k, crParamD + dDelta)
                crMinScore = mavScore(profile, committeeTupleToVector(K, profile.shape[1]))
                dDelta += 1
                if status:
                    copyK = K
            if status and bruteMinScore == crMinScore:           
                results.append((i, True, bruteMinScore == crMinScore))
            else:
                results.append((i,True,False))
        else:
            results.append((i, False, False))

    return results

def analyze(profile, k, d):
    print(profile, "\n")
    crSol = minimaxCR(profile, k, d)
    bruteSol = minimax(profile, k)
    print("Brute")
    pprint(bruteSol)
    print("\nCR")
    pprint(candsToLetters(crSol[1]))
    
#%%
d0 = list(filter(
    lambda i_stat_eq: i_stat_eq[1] == True and i_stat_eq[2] == False,
    compare(CR_44_A[:100], 2, 0)))

len(d0)

#%%
analyze(CR_44_A[14], 2, 1)

#%%
d0

#%%
CR_44_A[6]

#%%
minimaxCR(CR_44_A[6], 2, 2)