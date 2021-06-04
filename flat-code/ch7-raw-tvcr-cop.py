#%%
from definitions import Profile, Candidate, Voter
from vcrDetectionAlt import detectCRProperty, detectVRProperty, createGPEnv
from mavUtils import getVCRProfileInCROrder, getVCRProfileInVROrder, getVCRProfileInCRVROrder

#%%
import pandas as pd
import numpy as np
from itertools import combinations, product
from time import time, strftime, localtime
from typing import List, Tuple, NamedTuple
from functools import partial
import sys

#%%
### HELPERS

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

def detectFCOP(A, C, V, gEnv):
    return detectVRProperty(A, C, V, gEnv) and detectCRProperty(A, C, V, gEnv)

def detectCOP(A, C, V, gEnv):
    return detectVRProperty(A, C, V, gEnv) or detectCRProperty(A, C, V, gEnv)

#%%
### Single Deletion

class deletionSearchResults(NamedTuple):
    status:int
    k:int
    combination:List[int]

class fullDeletionSearch(NamedTuple):
    property:str
    axis:int
    k:int


def combinationDeletionSearch(A:np.ndarray, deleteAxis:int, propertyDetectionFun, gEnv) -> deletionSearchResults:
    deleteFunction = deleteCandidates if deleteAxis == 1 else deleteVoters
    k = 1
    results = []
    while k < 5:
        sT = time()
        for combination in map(list, combinations(range(A.shape[deleteAxis]), k)):
            tmpA = deleteFunction(A, combination)
            tmpV,tmpC = getVCLists(tmpA)
            propertyStatus = propertyDetectionFun(tmpA, tmpC, tmpV, gEnv)
            if propertyStatus:
                return deletionSearchResults(1, k, combination)

        k += 1
    print("NOT FOUND ", time() - sT)
    return deletionSearchResults(0, k, [])

def fullProfileSearch(profiles):
    env = createGPEnv()
    detectionFuns = {"cr":detectCRProperty, "vr":detectVRProperty}
    deleteAxes = {"cand":1, "voter":0}
    results = []
    for i,profile in enumerate(profiles):
        print("I={}, time2={}".format(i, strftime("%H:%M:%S", localtime())))
        for keyDel,deletionAxis in deleteAxes.items():
            minKCOP = 10000 # INF
            for keyProp,detectF in detectionFuns.items():
                res = combinationDeletionSearch(np.copy(profile.A), deletionAxis, detectF, env)
                minKCOP = min(minKCOP, res.k)
                res2 = fullDeletionSearch(property=keyProp, axis=keyDel, k=res.k)
                results.append(res2)
            results.append(fullDeletionSearch(property="cop", axis=keyDel, k=minKCOP))
    return results

#%%
### Multi Deletion
class multiDeletionSearchResults(NamedTuple):
    status:int
    kC:int
    kV:int
    combination:List[int]

class fullMultiDeletionSearch(NamedTuple):
    property:str
    axis:int
    kC:int
    kV:int

def multiCombinationDeletionSearch(A:np.ndarray, deleteAxis:int, propertyDetectionFun, gEnv) -> multiDeletionSearchResults:
    k = 1
    results = []
    sT = time()
    for dC,dV in sorted(list(product(range(1,5), range(1,5))), key=lambda t2: (t2[0] + t2[1])):
        for combC,combV in product(combinations(range(A.shape[1]), dC), combinations(range(A.shape[0]),dV)):
            tmpA = deleteCandidates(A, combC)
            tmpA = deleteVoters(tmpA, combV)
            tmpV,tmpC = getVCLists(tmpA)
            propertyStatus = propertyDetectionFun(tmpA, tmpC, tmpV, gEnv)
            if propertyStatus:
                return multiDeletionSearchResults(status=1, kC=dC, kV=dV, combination=[(combC, combV)])
    print("NOT FOUND ", time() - sT)
    return multiDeletionSearchResults(status=0, kC=dC+1, kV=dV+1, combination=[])

def fullProfileMultiSearch(profiles):
    env = createGPEnv()
    detectionFuns = {"cr":detectCRProperty, "vr":detectVRProperty}
    deleteAxes = {"cand":1, "voter":0}
    results = []
    for i,profile in enumerate(profiles):
        print("I={}, time2={}".format(i, strftime("%H:%M:%S", localtime())))
        for keyDel,deletionAxis in deleteAxes.items():
            minKCOP = (1000,1000)
            for keyProp,detectF in detectionFuns.items():
                res = multiCombinationDeletionSearch(np.copy(profile.A), deletionAxis, detectF, env)
                minKCOP = (res.kC, res.kV) if sum(minKCOP) > res.kC + res.kV else minKCOP
                res2 = fullMultiDeletionSearch(property=keyProp, axis=keyDel, kC=res.kC, kV=res.kV)
                results.append(res2)
            results.append(fullMultiDeletionSearch(property="cop", axis=keyDel, kC=minKCOP[0], kV=minKCOP[1]))
    return results


#%%
### RUNNER

def runner(start:int, end:int, C:int, V:int, distribution:str, R:int):
    print("RUNNING SINGLE DELETE")
    propertyType = "ncop"
    baseInPath = "resources/random/numpy/ncop/{}-{}-{}R-{}C{}V.npy".format(propertyType, distribution, R, C, V)
    baseOutStatsPath = "resources/random/ch7-transform/{}-{}-{}R-{}C{}V-{}S-{}E.csv".format("ncop", distribution, R, C, V, start, end)

    print("\nLoading from : {}\nSaving to : {}\n".format(baseInPath, baseOutStatsPath))

    profiles = map(Profile.fromNumpy, np.load(baseInPath)[start:end])

    transformStats = pd.DataFrame(fullProfileSearch(profiles))
    transformStats['distribution'] = distribution
    transformStats['R'] = R
    transformStats.to_csv(baseOutStatsPath, index=False, header=True)
    return transformStats

def runnerMulti(start:int, end:int, C:int, V:int, distribution:str, R:int):
    print("RUNNING MULTI DELETE")
    propertyType = "ncop"
    baseInPath = "resources/random/numpy/ncop/{}-{}-{}R-{}C{}V.npy".format(propertyType, distribution, R, C, V)
    baseOutStatsPath = "resources/random/ch7-transform/multi-{}-{}-{}R-{}C{}V-{}S-{}E.csv".format("ncop", distribution, R, C, V, start, end)

    print("\nLoading from : {}\nSaving to : {}\n".format(baseInPath, baseOutStatsPath))

    profiles = map(Profile.fromNumpy, np.load(baseInPath)[start:end])

    transformStats = pd.DataFrame(fullProfileMultiSearch(profiles))
    transformStats['distribution'] = distribution
    transformStats['R'] = R
    transformStats.to_csv(baseOutStatsPath, index=False, header=True)
    return transformStats

#%%
### MAIN

if __name__ == "__main__":
    C = 15
    V = 15
    s = int(sys.argv[1])
    e = int(sys.argv[2])
    distribution = sys.argv[3]
    R = int(sys.argv[4])
    sT = time()
    stats = runnerMulti(s, e, C, V, distribution, R)
    print(time() - sT)
    print(stats.describe())
