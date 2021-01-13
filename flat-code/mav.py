#%%
import numpy as np
from definitions import Profile, Candidate, Voter, VCR44
from itertools import combinations
from typing import List
import pickle


#%%
def committeeTupleToNp(committee:tuple, C:int) -> np.ndarray:
    committeeArray = np.zeros(C)
    np.add.at(committeeArray, list(committee), 1)
    return committeeArray
    
def hammingDistance(a: np.ndarray, b: np.ndarray) -> int:
    return (a != b).nonzero()[0].shape[0]

def mavScore(profile: Profile, committee: np.array) -> int:
    partialHD = lambda preference: hammingDistance(committee, preference)
    return max([partialHD(pref) for pref in profile.A])


def minimax(profile: Profile, k: int) -> np.array:
    candidateCount = len(profile.C)
    scores = [(mavScore(profile, committeeTupleToNp(committee, candidateCount)), [chr(ord('A') + i) for i in committee])
                for committee in combinations(range(candidateCount), k)]
    scores.sort(key=lambda mavScore_committee: mavScore_committee[0])
    return scores

#%%
P = VCR44()
print(P)

#%%
minimax(P,2)
