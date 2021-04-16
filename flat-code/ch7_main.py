#%%
from definitions import Profile, Candidate, Voter
from vcrDetectionAlt import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv
from mavUtils import getVCRProfileInCROrder, getVCRProfileInVROrder

import numpy as np
from itertools import combinations
from time import time
from typing import List, Tuple, NamedTuple

#%%
def VCRNCOP_44():
    A = np.load("resources/output/4C4V/NCOP-profiles/ncop-44-0.npy")
    return list(map(Profile.fromNumpy, A))

def VCRNCOP_55_1():
    A = np.load("resources/output/5C5V/NCOP-profiles/ncop-55-1.npy")
    return list(map(Profile.fromNumpy, A))

def VCRNCOP_55_2():
    A = np.load("resources/output/5C5V/NCOP-profiles/ncop-55-2.npy")
    return list(map(Profile.fromNumpy, A))

def VCRNCOP_55_3():
    A = np.load("resources/output/5C5V/NCOP-profiles/ncop-55-3.npy")
    return list(map(Profile.fromNumpy, A))

def VCRNCOP_66():
    A = np.load("resources/output/6C6V/NCOP-profiles/ncop-66-0.npy")
    return list(map(Profile.fromNumpy, A))

def VCRNCOP_1010():
    A = np.load("resources/output/10C10V/NCOP-profiles/ncop-1010-0.npy")
    return list(map(Profile.fromNumpy, A))

def CR_66_0():
    A = np.load("resources/output/6C6V/CR-profiles/cr-66-0.npy")
    return list(map(getVCRProfileInCROrder,map(Profile.fromNumpy, A)))

def VR_66_0():
    A = np.load("resources/output/6C6V/VR-profiles/vr-66-0.npy")
    return list(map(getVCRProfileInVROrder,map(Profile.fromNumpy, A)))

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
    status:bool
    k:int
    combination:List[int]

# axis=0 -> deleteVoters | axis=1 -> deleteCandidates
def combinationDeletionSearch(A:np.ndarray, deleteAxis:int, detectPartialFunction) -> deletionSearchResults:
    deleteFunction = deleteCandidates if deleteAxis == 1 else deleteVoters
    crProfiles = [] 
    found = False
    k = 1
    while not found:
        for combination in map(list, combinations(range(A.shape[deleteAxis]), k)):
            tmpA = deleteFunction(A, combination)
            Vs, Cs = getVCLists(tmpA)
            # if detectCRProperty(tmpA, Cs, Vs, gEnv):
            if detectPartialFunction(tmpA, Cs, Vs):
                return deletionSearchResults(True, k, combination)
        k += 1
    return deletionSearchResults(False, k, [])

def exampleRun():
    As = np.load("resources/output/10C10V/NCOP-profiles/ncop-1010-0.npy")
    P1010 = list(map(Profile.fromNumpy, As))
    gEnv = createGPEnv()
    detectPF = partial(detectCRPropertyWrapper, gEnv)
    a = P1010[19].A
    combinationDeletionSearch(P1010[11].A, 1, detectPF)