#%%
from definitions import Profile, Candidate, Voter
from vcrDetectionAlt import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv

import numpy as np
from itertools import combinations
from time import time
from typing import List, Tuple

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

#%%
P66 = VCRNCOP_66()
P1010 = VCRNCOP_1010() # 1, 11, 19, 22 nie maja k=1

#%%


#%%
gEnv = createGPEnv()

#%%
# axis=0 -> deleteVoters | axis=1 -> deleteCandidates
def combinationDeletionSearch(A:np.ndarray, axis):
    deleteFunction = deleteCandidates if axis == 1 else deleteVoters
    crProfiles = [] 
    found = False
    k = 1
    start = time()
    while not found:
        print("K={}".format(k))
        for comb in map(list,combinations(range(A.shape[axis]), k)):
            tmpA = deleteFunction(A, comb)
            Vs, Cs = getVCLists(tmpA)
            if detectCRProperty(tmpA, Cs, Vs, gEnv):
                crProfiles.append((comb, tmpA))
                found = True
                break
        k += 1

    print(time() - start)
    return crProfiles

#%%
a = np.array(P1010[19].A)
combinationDeletionSearch(a, 0)

#%%
crA = crProfiles[0][1]
crVs, crCs = getVCLists(crA)
detectCRProperty(crA, crCs, crVs, gEnv)

#%%
all((i == 1 for i in range(4)))