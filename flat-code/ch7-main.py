#%%
from definitions import Profile, Candidate, Voter
from vcrDetectionAlt import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv

import numpy as np
from itertools import combinations
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

#%%
P66 = VCRNCOP_66()

#%%
a = np.array(P66[1].A)
a

#%%
gEnv = createGPEnv()

#%%
k = 1
crProfiles = [] 

for comb in map(list,combinations(range(a.shape[1]), k)):
    tmpA = deleteCandidates(A=a, candidates=comb)
    Vs, Cs = getVCLists(tmpA)
    if detectCRProperty(tmpA, Cs, Vs, gEnv):
        crProfiles.append((comb, tmpA))
        
crProfiles

#%%
crProfiles[0]

#%%
crA = crProfiles[0][1]
crVs, crCs = getVCLists(crA)
detectCRProperty(crA, crCs, crVs, gEnv)
