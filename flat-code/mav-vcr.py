#%%
from functools import partial

import numpy as np
from numpy import ndarray

from definitions import Profile
from mavUtils import getVCRProfileInCROrder, getVCRProfileInVROrder, getVCRProfileInCRVROrder
from mavUtils import mavScore, basePartialCompare
from mavUtils import minimax, minimaxFull, minimaxCR, STATUS_FAIL


#%%
def VCRNCOP_44():
    return np.load("resources/output/4C4V/NCOP-profiles/ncop-44-0.npy")

def VCRNCOP_55_1():
    return np.load("resources/output/5C5V/NCOP-profiles/ncop-55-1.npy")

def VCRNCOP_55_2():
    return np.load("resources/output/5C5V/NCOP-profiles/ncop-55-2.npy")

def VCRNCOP_55_3():
    return np.load("resources/output/5C5V/NCOP-profiles/ncop-55-3.npy")

def VCRNCOP_66():
    return np.load("resources/output/6C6V/NCOP-profiles/ncop-66-0.npy")

def bruteMAVWrapper(k:int, A:np.ndarray):
    return minimax(A=A, k=k)

def crMAVWrapper(k:int, d:int, A:np.ndarray):
    return minimaxCR(A=A, k=k, d=d)

def singleCompareMAVs(A, k, d):
    bruteMAVPartial = partial(bruteMAVWrapper, k)
    crMAVPartial = partial(crMAVWrapper, k, d)
    return basePartialCompare(A=A, algoX=bruteMAVPartial, algoY=crMAVPartial)

def compareMAVs(profiles, k, d):
    results = [singleCompareMAVs(profile.A, k, d) for profile in profiles]
    falseResults = {i:result for i,result in enumerate(results) if result.status == STATUS_FAIL}
    return falseResults

def parameterCompareMAVs(profiles, k, dRange):
    S = [set(compareMAVs(profiles=profiles, k=k, d=d).keys()) for d in dRange]
    return S[0].intersection(*S)

#%%
P44 = list(map(Profile.fromNumpy, VCRNCOP_44()))
P55_1 = list(map(Profile.fromNumpy, VCRNCOP_55_1()))
P55_2 = list(map(Profile.fromNumpy, VCRNCOP_55_2()))
P55_3 = list(map(Profile.fromNumpy, VCRNCOP_55_3()))
P66 = list(map(Profile.fromNumpy, VCRNCOP_66()))

CR_P44 = list(map(getVCRProfileInCROrder, map(Profile.fromNumpy, VCRNCOP_44())))
CR_P55_1 = list(map(getVCRProfileInCROrder, map(Profile.fromNumpy, VCRNCOP_55_1())))
CR_P55_2 = list(map(getVCRProfileInCROrder, map(Profile.fromNumpy, VCRNCOP_55_2())))
CR_P55_3 = list(map(getVCRProfileInCROrder, map(Profile.fromNumpy, VCRNCOP_55_3())))
CR_P66 = list(map(getVCRProfileInCROrder, map(Profile.fromNumpy, VCRNCOP_66())))

VR_P44 = list(map(getVCRProfileInVROrder, map(Profile.fromNumpy, VCRNCOP_44())))
VR_P55_1 = list(map(getVCRProfileInVROrder, map(Profile.fromNumpy, VCRNCOP_55_1())))
VR_P55_2 = list(map(getVCRProfileInVROrder, map(Profile.fromNumpy, VCRNCOP_55_2())))
VR_P55_3 = list(map(getVCRProfileInVROrder, map(Profile.fromNumpy, VCRNCOP_55_3())))
VR_P66 = list(map(getVCRProfileInVROrder, map(Profile.fromNumpy, VCRNCOP_66())))

CR_VR_P44 = list(map(getVCRProfileInCRVROrder, map(Profile.fromNumpy, VCRNCOP_44())))
CR_VR_P55_1 = list(map(getVCRProfileInCRVROrder, map(Profile.fromNumpy, VCRNCOP_55_1())))
CR_VR_P55_2 = list(map(getVCRProfileInCRVROrder, map(Profile.fromNumpy, VCRNCOP_55_2())))
CR_VR_P55_3 = list(map(getVCRProfileInCRVROrder, map(Profile.fromNumpy, VCRNCOP_55_3())))
CR_VR_P66 = list(map(getVCRProfileInCRVROrder, map(Profile.fromNumpy, VCRNCOP_66())))

#%%
# P55 base | CR | VR
# k=2 : {80} | {} | {}
# k=3 : {8,9,45,73} | {32, 73} | {8, 9, 47, 70, 73}
# k=4 : {49, 86} | {35} | {16, 18, 84, 86}

# P55_2
# k=1: {} | {} | {13}
# k=2 : {} | {} | {}
# k=3 : {} | {} | {}
# k=4 : {5, 19} | {5} | {24}
# k=5 : {11} | {9, 11} | {11}

# P55_3
# k=2 : {21} | {} | {12}
# k=3 : {} | {} | {}
# k=4 : {5, 18, 22} | {} | {5, 22}

# P66
# k=1 : {} | {28, 50, 172} | {84, 96}
# k=2 : {27, 53, 90, 101, 161, 166} | {50, 54, 91, 119} | {113, 131}
# k=3 : {58, 163, 168} | {54, 177} | {33, 58, 80, 106, 163, 173}
# k=4 : {27, 40, 112, 128, 150, 158, 168} | {36, 40, 45, 54, 73, 111, 128} | {4, 112, 128}


#%%
parameterCompareMAVs(profiles=P55_1, k=3, dRange=range(6))

#%%
mavScore(CR_P55_1[73].A, [1,1,0,1,0])

#%%
compareMAVs([P55_1[73]], k=3, d=5)

#%%
print(CR_P55_1[73])

#%%
A = np.array([4,5,6,7])
A[[0,1,2,3]] = A[[2,3,1,0]]
A

#%%
minimaxFull(A=P55_1[0].A, k=2)

#%%
minimaxCR(A=CR_P55_1[].A, k=3, d=2)

#%%
def getCandsSortedByVoteFromVoter(A:ndarray, X:list, v:int) -> list:
    candidateVoteCount = [(c, sum(A[v:,c])) for c in range(A.shape[1])]
    candidateVoteCount.sort(key=lambda t: -t[1])
    return [t[0] for t in candidateVoteCount if t[0] in X]

#%%
print(a)
getCandsSortedByVoteFromVoter(a, [0,1,2], 0)

#%%
print(vCands)

#%%
a = np.array(P55_1[73].A)
cVotes = [(c, sum(a[0:,c])) for c in range(a.shape[1])]
vCands = [(v, sum(a[v])) for v in range(a.shape[0])]
sorted(cVotes, key=lambda t: (-t[1], vCands[t[0]]))

#%%
print(P55_1[73])