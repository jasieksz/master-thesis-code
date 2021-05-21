#%%
from definitions import Profile, Candidate, Voter
from vcrDetectionAlt import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv
from vcrDetectionAlt import findCRPoints, findVRPoints
from mavUtils import getVCRProfileInCROrder, getVCRProfileInVROrder, getVCRProfileInCRVROrder

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import combinations
from time import time
from typing import List, Tuple, NamedTuple
from functools import partial


#%%
# delete columns
def deleteCandidates(A:np.ndarray, candidates:List[int]) -> np.ndarray:
    return np.delete(A, obj=candidates, axis=1)

# delete rows
def deleteVoters(A:np.ndarray, voters:List[int]) -> np.ndarray:
    return np.delete(A, obj=voters, axis=0)

def getVCLists(A:np.ndarray) -> Tuple[List[str],List[str]]:
    V = ['v' + str(i) for i in range(A.shape[0])]
    C = ['c' + str(i) for i in range(A.shape[1])]
    return V,C

def detectCRPropertyWrapper(gEnv, A, C, V):
    return detectCRProperty(A, C, V, gEnv)

def detectVRPropertyWrapper(gEnv, A, C, V):
    return detectVRProperty(A, C, V, gEnv)

class deletionSearchResults(NamedTuple):
    status:int
    k:int
    combination:List[int]


def fullDetectProperty(A:np.ndarray, env) -> int:
    Vs, Cs = getVCLists(A)
    crResult = detectCRProperty(A=A, C=Cs, V=Vs, env=env)
    vrResult = detectVRProperty(A=A, C=Cs, V=Vs, env=env)
    status = 0
    if (crResult and not vrResult):
        status = 1
    elif (not crResult and vrResult):
        status = 2
    elif (crResult and vrResult):
        status = 3
    return status

# axis=0 -> deleteVoters | axis=1 -> deleteCandidates
def combinationDeletionSearch(A:np.ndarray, deleteAxis:int, gEnv) -> deletionSearchResults:
    deleteFunction = deleteCandidates if deleteAxis == 1 else deleteVoters
    found = False
    k = 1
    results = []
    while not found and k < 5:
        for combination in map(list, combinations(range(A.shape[deleteAxis]), k)):
            tmpA = deleteFunction(A, combination)
            propertyStatus = fullDetectProperty(tmpA, gEnv)
            if propertyStatus != 0:
                found = True
                results.append(deletionSearchResults(propertyStatus, k, combination))
        k += 1
    return results if found else deletionSearchResults(0, k, [])

def exampleRun(profile):
    env = createGPEnv()


def NCOP_1010():
    return list(map(Profile.fromNumpy, np.load("resources/random/numpy/ncop-2gauss-8R-10C10V-0S-100E.npy")))

def prefHeatmap(profile):
    plt.figure(figsize=(6,5))
    sns.heatmap((getVCRProfileInCRVROrder(profile)).A, cmap=['black', 'gray'])
    plt.show()

#############################################
#############################################
# NOTEBOOK
#%%
P1010 = NCOP_1010()
env = createGPEnv()

#%%
# prefHeatmap(P1010[5])
res = combinationDeletionSearch(P1010[5].A, 1, env)
res

#%%
tmpA = deleteCandidates(np.copy(P1010[5].A), [6])
vs,cs = getVCLists(tmpA)
_,res = findCRPoints(tmpA, cs, vs, env)
tmpP = Profile.fromILPRes(tmpA, res, cs, vs)
prefHeatmap(tmpP)
