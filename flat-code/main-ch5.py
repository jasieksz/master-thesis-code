#%%
from vcrDetectionAlt import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv
from vcrDomain import isVCR, vcrPropertyRaw
from definitions import Profile
from mavUtils import getVCRProfileInCRVROrder
from matplotlib import pyplot as plt

#%%
import os
import numpy as np
from numpy.random import default_rng
import pandas as pd
import seaborn as sns
from time import time
from typing import List, Tuple, Dict
from functools import partial


#%%
# HELPERS
##################################################
##################################################
def getVCLists(A:np.ndarray):
    V = ['v' + str(i) for i in range(A.shape[0])]
    C = ['c' + str(i) for i in range(A.shape[1])]
    return V,C

def mergeDictLists(d1, d2):
  keys = set(d1).union(d2)
  no = []
  return dict((k, d1.get(k, no) + d2.get(k, no)) for k in keys)

def npProfileFromVotersAndCandidates(voters: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    A = np.zeros((len(voters),len(candidates)))
    for vI, (vX,vR) in enumerate(voters):
        for cI, (cX,cR) in enumerate(candidates):
            if vcrPropertyRaw(vX, vR, cX, cR):
                A[vI,cI] = 1
    
    return np.concatenate([
            np.array(A.shape),
            candidates.flatten(),
            voters.flatten(),
            A.flatten()])    


def uniformRadiusWrapper(RNG, low, high, size):
    return RNG.uniform(low=low, high=high, size=size)

def gaussRadiusWrapper(RNG, mean, std, size):
    return RNG.normal(mean, std, size)

def normalizeRadius(radii:np.ndarray) -> np.ndarray:
    m = np.min(radii)
    if m < 0:
        radii -= m
    return radii


#%%
# UNIFORM
##################################################
##################################################

def generateVCRProfileByRadiusUniform(RNG,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> Profile:

    resultProfiles = {k:list() for k in radiusParams.keys()}
    xMin = -10
    xMax = 10

    for key, (rCFun,rVFun) in radiusParams.items():
        radiiC = normalizeRadius(rCFun(C))
        radiiV = normalizeRadius(rVFun(V))

        cPositions = RNG.uniform(low=xMin, high=xMax, size=C)
        vPositions = RNG.uniform(low=xMin, high=xMax, size=V)

        candidates = np.dstack((cPositions, radiiC))[0]
        voters = np.dstack((vPositions, radiiV))[0]
        profile = npProfileFromVotersAndCandidates(voters, candidates)
        resultProfiles[key].append(profile)
    
    return resultProfiles

def generateVCRProfilesByRadiusUniform(RNG, count:int,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> List[Profile]:
    resultProfiles = {k:list() for k in radiusParams.keys()}

    for i in range(count):
        profiles = generateVCRProfileByRadiusUniform(RNG=R, C=C, V=V, radiusParams=radiusParams)
        resultProfiles = mergeDictLists(resultProfiles, profiles)

    return resultProfiles

def runnerVCRProfilesByRadiusUniform(C:int, V:int):
    RNG=default_rng()
    distribution = 'uniform'
    count = 1000

    radiusParams={
        4:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        5:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        6:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        7:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        8:(partial(uniformRadiusWrapper, RNG, 0, 3), partial(uniformRadiusWrapper, RNG, 0, 3)),
    }

    path = "resources/random/numpy/vcr-{}-{}R-{}C{}V.npy"

    profilesByR = generateVCRProfilesByRadiusUniform(RNG, count, C, V, radiusParams)

    for rParam, profiles in profilesByR.items():
        saveLoc = path.format(distribution, rParam, C, V)
        print("Saving to : {}".format(saveLoc))
        with open(saveLoc, 'wb') as f:
            np.save(file=f, arr=profiles, allow_pickle=False)


#%%
# 2 GAUSS
##################################################
##################################################
#%%
def generateVCRProfileByRadius2Gauss(RNG,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> Profile:

    resultProfiles = {k:list() for k in radiusParams.keys()}

    majorityC = int(C * 0.7)
    minorityC = C - majorityC
    majorityV = int(V * 0.7)
    minorityV = V - majorityV

    for key, (rCFun,rVFun) in radiusParams.items():
        radiiC = normalizeRadius(rCFun(C))
        radiiV = normalizeRadius(rVFun(V))

        positionsCMajor = RNG.normal(-3, 1.8, size=majorityC)
        positionsCMinor = RNG.normal(3, 1.8, size=minorityC)
        cPositions = np.append(positionsCMajor, positionsCMinor)

        positionsVMajor = RNG.normal(-3, 1.8, size=majorityV)
        positionsVMinor = RNG.normal(3, 1.8, size=minorityV)
        vPositions = np.append(positionsVMajor, positionsVMinor)

        candidates = np.dstack((cPositions, radiiC))[0]
        voters = np.dstack((vPositions, radiiV))[0]
        profile = npProfileFromVotersAndCandidates(voters, candidates)
        resultProfiles[key].append(profile)
    
    return resultProfiles

def generateVCRProfilesByRadius2Gauss(RNG, count:int,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> List[Profile]:
    resultProfiles = {k:list() for k in radiusParams.keys()}

    for i in range(count):
        profiles = generateVCRProfileByRadius2Gauss(RNG=R, C=C, V=V, radiusParams=radiusParams)
        resultProfiles = mergeDictLists(resultProfiles, profiles)

    return resultProfiles

def runnerVCRProfilesByRadius2Gauss(C:int, V:int):
    RNG=default_rng()
    distribution = '2gauss'
    count = 1000

    radiusParams={
        4:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        5:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        6:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        7:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        8:(partial(uniformRadiusWrapper, RNG, 0, 3), partial(uniformRadiusWrapper, RNG, 0, 3)),
    }

    path = "resources/random/numpy/vcr-{}-{}R-{}C{}V.npy"

    profilesByR = generateVCRProfilesByRadius2Gauss(RNG, count, C, V, radiusParams)

    for rParam, profiles in profilesByR.items():
        saveLoc = path.format(distribution, rParam, C, V)
        print("Saving to : {}".format(saveLoc))
        with open(saveLoc, 'wb') as f:
            np.save(file=f, arr=profiles, allow_pickle=False)

#%%
# Gauss Uniform
##################################################
##################################################
def generateVCRProfileByRadiusGaussUniform(RNG,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> Profile:

    resultProfiles = {k:list() for k in radiusParams.keys()}

    majorityC = int(C * 0.7)
    minorityC = C - majorityC

    for key, (rCFun,rVFun) in radiusParams.items():
        radiiC = normalizeRadius(rCFun(C))
        radiiV = normalizeRadius(rVFun(V))

        positionsCMajor = RNG.normal(-3, 1.8, size=majorityC)
        positionsCMinor = RNG.normal(3, 1.8, size=minorityC)
        cPositions = np.append(positionsCMajor, positionsCMinor)
        vPositions = RNG.uniform(low=-10, high=10, size=V)
        
        candidates = np.dstack((cPositions, radiiC))[0]
        voters = np.dstack((vPositions, radiiV))[0]
        profile = npProfileFromVotersAndCandidates(voters, candidates)
        resultProfiles[key].append(profile)
    
    return resultProfiles

def generateVCRProfilesByRadiusGaussUniform(RNG, count:int,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> List[Profile]:
    resultProfiles = {k:list() for k in radiusParams.keys()}

    for i in range(count):
        profiles = generateVCRProfileByRadiusGaussUniform(RNG=R, C=C, V=V, radiusParams=radiusParams)
        resultProfiles = mergeDictLists(resultProfiles, profiles)

    return resultProfiles

def runnerVCRProfilesByRadiusGaussUniform(C:int, V:int):
    RNG=default_rng()
    distribution = 'gaussuniform'
    count = 1000
    radiusParams={
        4:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        5:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        6:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        7:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        8:(partial(uniformRadiusWrapper, RNG, 0, 3), partial(uniformRadiusWrapper, RNG, 0, 3)),
    }

    path = "resources/random/numpy/vcr-{}-{}R-{}C{}V.npy"

    profilesByR = generateVCRProfilesByRadiusGaussUniform(RNG, count, C, V, radiusParams)

    for rParam, profiles in profilesByR.items():
        saveLoc = path.format(distribution, rParam, C, V)
        print("Saving to : {}".format(saveLoc))
        with open(saveLoc, 'wb') as f:
            np.save(file=f, arr=profiles, allow_pickle=False)

#%%
# Uniform Gauss
##################################################
##################################################
def generateVCRProfileByRadiusUniformGauss(RNG,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> Profile:

    resultProfiles = {k:list() for k in radiusParams.keys()}

    majorityV = int(C * 0.7)
    minorityV = V - majorityV

    for key, (rCFun,rVFun) in radiusParams.items():
        radiiC = normalizeRadius(rCFun(C))
        radiiV = normalizeRadius(rVFun(V))

        cPositions = RNG.uniform(low=-10, high=10, size=V)
        positionsVMajor = RNG.normal(-3, 1.8, size=majorityV)
        positionsVMinor = RNG.normal(3, 1.8, size=minorityV)
        vPositions = np.append(positionsVMajor, positionsVMinor)

        candidates = np.dstack((cPositions, radiiC))[0]
        voters = np.dstack((vPositions, radiiV))[0]
        profile = npProfileFromVotersAndCandidates(voters, candidates)
        resultProfiles[key].append(profile)
    
    return resultProfiles

def generateVCRProfilesByRadiusUniformGauss(RNG, count:int,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> List[Profile]:
    resultProfiles = {k:list() for k in radiusParams.keys()}

    for i in range(count):
        profiles = generateVCRProfileByRadiusUniformGauss(RNG=R, C=C, V=V, radiusParams=radiusParams)
        resultProfiles = mergeDictLists(resultProfiles, profiles)

    return resultProfiles

def runnerVCRProfilesByRadiusUniformGauss(C:int, V:int):
    RNG=default_rng()
    distribution = 'uniformgauss'
    count = 1000
    radiusParams={
        4:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        5:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        6:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        7:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        8:(partial(uniformRadiusWrapper, RNG, 0, 3), partial(uniformRadiusWrapper, RNG, 0, 3)),
    }
    path = "resources/random/numpy/vcr-{}-{}R-{}C{}V.npy"

    profilesByR = generateVCRProfilesByRadiusUniformGauss(RNG, count, C, V, radiusParams)

    for rParam, profiles in profilesByR.items():
        saveLoc = path.format(distribution, rParam, C, V)
        print("Saving to : {}".format(saveLoc))
        with open(saveLoc, 'wb') as f:
            np.save(file=f, arr=profiles, allow_pickle=False)

##################################################
# NOTEBOOK
##################################################

#%%
R = default_rng()
gEnv = createGPEnv()

#%%
runnerVCRProfilesByRadiusUniform(40,40)

#%%
runnerVCRProfilesByRadius2Gauss(40,40)

#%%
runnerVCRProfilesByRadiusGaussUniform(40,40)

#%%
runnerVCRProfilesByRadiusUniformGauss(40,40)


#%%
i = 2
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))
sns.heatmap((getVCRProfileInCRVROrder(Profile.fromNumpy(d[0][i])).A), cmap=['black', 'gray'], ax=ax1)
sns.heatmap((getVCRProfileInCRVROrder(Profile.fromNumpy(d[1][i])).A), cmap=['black', 'gray'], ax=ax2)
ax1.set_title("Initial Approval Matrix")
ax1.set_xlabel("candidates")
ax1.set_ylabel("voters")

ax2.set_title("VCR Ordered Approval Matrix")
ax2.set_xlabel("candidates (reindexed)")
ax2.set_ylabel("voters (reindexed)")

#%%
P44_0 = np.load('resources/random/numpy/vcr-uniform-0R-100C100V.npy')
P44_1 = np.load('resources/random/numpy/vcr-uniform-1R-100C100V.npy')
P44_2 = np.load('resources/random/numpy/vcr-uniform-2R-100C100V.npy')
P44_3 = np.load('resources/random/numpy/vcr-uniform-3R-100C100V.npy')

#%%
print(Profile.fromNumpy(P44_3[0]))

#%%
i = 6
fig, (ax1, ax2) = plt.subplots(2,2, figsize=(14,12))
sns.heatmap(getVCRProfileInCRVROrder(Profile.fromNumpy(P44_0[i])).A, cmap=['black', 'gray'], ax=ax1[0])
sns.heatmap(getVCRProfileInCRVROrder(Profile.fromNumpy(P44_1[i])).A, cmap=['black', 'gray'], ax=ax1[1])
sns.heatmap(getVCRProfileInCRVROrder(Profile.fromNumpy(P44_2[i])).A, cmap=['black', 'gray'], ax=ax2[0])
sns.heatmap(getVCRProfileInCRVROrder(Profile.fromNumpy(P44_3[i])).A, cmap=['black', 'gray'], ax=ax2[1])
ax1[0].set_title("R=0.7")
ax1[0].set_xlabel("candidates")
ax1[0].set_ylabel("voters")

ax1[1].set_title("R=1.2")
ax1[1].set_xlabel("candidates")
ax1[1].set_ylabel("voters")

ax2[0].set_title("R=Uniform<0.7,1.2>")
ax2[0].set_xlabel("candidates")
ax2[0].set_ylabel("voters")

ax2[1].set_title("R=Uniform<0,3>")
ax2[1].set_xlabel("candidates")
ax2[1].set_ylabel("voters")

#%%
i = 6
fig, (ax1, ax2) = plt.subplots(2,2, figsize=(14,12))
sns.heatmap((Profile.fromNumpy(P44_0[i])).A, cmap=['black', 'gray'], ax=ax1[0])
sns.heatmap((Profile.fromNumpy(P44_1[i])).A, cmap=['black', 'gray'], ax=ax1[1])
sns.heatmap((Profile.fromNumpy(P44_2[i])).A, cmap=['black', 'gray'], ax=ax2[0])
sns.heatmap((Profile.fromNumpy(P44_3[i])).A, cmap=['black', 'gray'], ax=ax2[1])
ax1[0].set_title("R=0.7")
ax1[0].set_xlabel("candidates")
ax1[0].set_ylabel("voters")

ax1[1].set_title("R=1.2")
ax1[1].set_xlabel("candidates")
ax1[1].set_ylabel("voters")

ax2[0].set_title("R=Uniform<0.7,1.2>")
ax2[0].set_xlabel("candidates")
ax2[0].set_ylabel("voters")

ax2[1].set_title("R=Uniform<0,3>")
ax2[1].set_xlabel("candidates")
ax2[1].set_ylabel("voters")


#%%
# vIds, cIds = getVCLists(Profile.fromNumpy(P44_3[6]).A) 
sT = time()
print(detectVRProperty(Profile.fromNumpy(P44_3[6]).A, cIds, vIds, gEnv))
time() - sT


#%%

#%%
radiusParams={
    4:"G(1.5,0.5) G(1.5,0.5)",
    5:"U(0.7,1.2) G(1.5,0.5)",
    6:"G(1.5,0.5) U(0.7,1.2)",
    7:"U(0.7,1.2) U(0.7,1.2)",
    8:"U(0,3) U(0,3)"
}


def f():
    for dist in ['uniform','2gauss','uniformgauss','gaussuniform']:
        for r in range(4,9):
            path = [e for e in os.listdir("resources/random/spark/20C20V/ncop-{}-{}R-stats/".format(dist, r)) if e[-3:] == "csv"][0]
            df = pd.read_csv("resources/random/spark/20C20V/ncop-{}-{}R-stats/{}".format(dist, r, path))
            dist2 = dist
            if dist == 'gaussuniform':
                dist2 = 'gauss C\n uniform V'
            if dist == 'uniformgauss':
                dist2 = 'uniform C\n gauss V'
            df['distribution'] = dist2
            df['R'] = radiusParams[r]
            yield df

#%%
data = pd.concat(f())

#%%

g = sns.catplot(data=data, x='distribution', y='count',
    hue='property', col='R', col_wrap=3,
    orient="v", kind='bar', sharex=False, sharey=False)

g.fig.subplots_adjust(top=0.9)
for ax in g.axes:
    ax.set_xlabel('')
g.fig.suptitle('20 Candidates 20 Voters')

#%%
data = pd.read_csv("resources/output/merged-stats-small-vcr.csv")

#%%
g = sns.catplot(data=data, x='property', y='count',
    col='election', col_wrap=3,
    orient="v", kind='bar', sharex=False, sharey=False, log=True)

g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Small Size Elections - All Combinations')
