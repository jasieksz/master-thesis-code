# %%
import numpy as np
from profiles.definitions import Profile, Candidate, Voter
from itertools import combinations
from typing import List
import pickle

# %%
a = np.array([1, 1, 0, 0])
b = np.array([0, 1, 1, 0])


# %%
def hammingDistance(a: np.array, b: np.array) -> int:
    return (a != b).nonzero()[0].shape[0]


def mavScore(profile: Profile, committee: np.array) -> int:
    partialHD = lambda preference: hammingDistance(committee, preference)
    return max([partialHD(pref) for pref in profile.A])


def minimax(profile: Profile, k: int) -> np.array:
    candidateCount = len(profile.C)
    scores = [(mavScore(profile, np.array(committee)), profile) for committee in combinations(range(candidateCount), k)]
    scores.sort(key=lambda mav_p: mav_p[0])
    return scores


# %%
#    A  B  C  D
# v0  1  1  1  1
# v1  1  1  0  0
# v2  0  1  0  1
# v3  0  1  1  0

# %%
c = 4
v = 4
basePath = '../profiles/profiles-{}C-{}V/'.format(c, v)
allPath = 'all-P{}{}.npy'.format(c, v)
vcrPath = 'VCR-P{}{}'.format(c, v)
vcrFNCOPPath = 'VCR-NCOP-P{}{}'.format(c, v)

with open(basePath + vcrFNCOPPath, 'rb') as f:
    vcrFullNCOP_P44 = pickle.load(f)
print(len(vcrFullNCOP_P44))

# %%
# [(s, p.A) for s,p in minimax(vcrFullNCOP_P44[0], 1)]
