#%%
from vcrDetectionAlt import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv
from vcrDomain import isVCR, vcrPropertyRaw
from definitions import Profile
from mavUtils import getVCRProfileInCRVROrder
from matplotlib import pyplot as plt

import numpy as np
from numpy.random import default_rng
import pandas as pd
import seaborn as sns
from time import time
from typing import List, Tuple, Dict

#%%
def generateDoubleGaussRandomAgents(RNG, count:int, proportion:float, \
        meanMajor:float, meanMinor:float, std:float, \
        radiusConst:float) -> np.ndarray: 
    
    majority = int(count * proportion)
    minority = count - majority

    positionsMajor = RNG.normal(meanMajor, std, size=majority)
    positionsMinor = RNG.normal(meanMinor, std, size=minority)

    radii = np.ones(count) * radiusConst

    positions = np.append(positionsMajor, positionsMinor)
    return np.dstack((positions, radii))[0]

def generateUniformRandomAgents(RNG, count:int,
        rMin:int, rMax:int,
        xMin:int, xMax:int) -> np.ndarray:
    positions = RNG.uniform(low=xMin, high=xMax, size=count)
    radii = RNG.uniform(low=rMin, high=rMax, size=count)
    return np.dstack((positions, radii))[0]

#%%
def generateRandomVCRProfile(RNG, C:int, V:int,
        proportionC:float, proportionV:float,
        meanMajor:float, meanMinor:float, std:float, \
        radiusConst:float) -> Profile:
    candidates = generateUniformRandomAgents(RNG=RNG, count=C, rMin=0, rMax=1, xMin=-4, xMax=4)

    # candidates = generateDoubleGaussRandomAgents(RNG=RNG, count=V, proportion=proportionC,
    #     meanMajor=meanMajor, meanMinor=meanMinor, std=std, radiusConst=radiusConst)

    voters = generateUniformRandomAgents(RNG=RNG, count=V, rMin=0, rMax=1, xMin=-4, xMax=4)


    # voters = generateDoubleGaussRandomAgents(RNG=RNG, count=V, proportion=proportionV,
    #     meanMajor=meanMajor, meanMinor=meanMinor, std=std, radiusConst=radiusConst)

    A = np.zeros((V,C))
    for vI, (vX,vR) in enumerate(voters):
        for cI, (cX,cR) in enumerate(candidates):
            if vcrPropertyRaw(vX, vR, cX, cR):
                A[vI,cI] = 1
    
    npProfile = np.concatenate([
            np.array(A.shape),
            candidates.flatten(),
            voters.flatten(),
            A.flatten()])    

    return Profile.fromNumpy(npProfile)

#%%
def generateRandomVCRProfile(RNG, C:int, V:int,
     agentRandomFunction: Callable[[int,int,int,int,int], np.ndarray]) -> Profile:
    
    rConstMin = 0.7
    rConstMax = 1.2
    candidates = agentRandomFunction(RNG=RNG, count=C, rMin=rConstMin, rMax=rConstMax, xMin=-10, xMax=10)
    voters = agentRandomFunction(RNG, V, rConstMin, rConstMax, -10, 10)

    A = np.zeros((V,C))
    for vI, (vX,vR) in enumerate(voters):
        for cI, (cX,cR) in enumerate(candidates):
            if vcrPropertyRaw(vX, vR, cX, cR):
                A[vI,cI] = 1
    
    npProfile = np.concatenate([
            np.array(A.shape),
            candidates.flatten(),
            voters.flatten(),
            A.flatten()])    

    return Profile.fromNumpy(npProfile)


#%%
def getVCLists(A:np.ndarray):
    V = ['v' + str(i) for i in range(A.shape[0])]
    C = ['c' + str(i) for i in range(A.shape[1])]
    return V,C

#%%
R = default_rng()
gEnv = createGPEnv()


#%%
# P = generateRandomVCRProfile(RNG=R, C=20, V=20,
#         proportionC=0.7, proportionV=0.7,
#         meanMajor=-1.5, meanMinor=1.5, std=0.8,
#         radiusConst=0.7)

startTime = time()
P = generateRandomVCRProfile(R, 20, 20, generateUniformRandomAgents)
print(time() - startTime)

startTime = time()
sns.heatmap(P.A, cmap=['black', 'gray'])
plt.show()
sns.heatmap(getVCRProfileInCRVROrder(P).A, cmap=['black', 'gray'])
vIds, cIds = getVCLists(P.A) 
print(time() - startTime)

startTime = time()
cr = detectCRProperty(P.A, cIds, vIds, gEnv)
vr = detectVRProperty(P.A, cIds, vIds, gEnv)
print(time() - startTime)
startTime = time()

print(cr, vr)

#%%
cOrdered = [int(c.id[1:]) for c in sorted(list(P.C), key=lambda c: c.x)]
vOrdered = [int(v.id[1:]) for v in sorted(list(P.V), key=lambda v: v.x)]


#%%
print()

#%%
c = np.empty(0)

#%%
a = np.append(R.normal(3,1.8,70000),R.normal(-3,1.8,30000))
df = pd.DataFrame(a, columns=['x'])
sns.displot(data=df, x='x')

#%%
b = np.append(R.normal(3,1.8,700),R.normal(-3,1.8,300))
c = np.append(b,c)
df2 = pd.DataFrame(c, columns=['x'])
sns.displot(data=df2, x='x')


#%%

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

def generateVCRProfileByRadius(RNG,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> Profile:

    resultProfiles = {k:list() for k in radiusParams.keys()}
    xMin = -10
    xMax = 10

    cPositions = RNG.uniform(low=xMin, high=xMax, size=C)
    vPositions = RNG.uniform(low=xMin, high=xMax, size=C)

    for key, (rMin,rMax) in radiusParams.items():
        radiiC = RNG.uniform(low=rMin, high=rMax, size=C)
        radiiV = RNG.uniform(low=rMin, high=rMax, size=C)
        candidates = np.dstack((cPositions, radiiC))[0]
        voters = np.dstack((vPositions, radiiV))[0]
        profile = npProfileFromVotersAndCandidates(voters, candidates)
        resultProfiles[key].append(profile)
    
    return resultProfiles

def mergeDictLists(d1, d2):
  keys = set(d1).union(d2)
  no = []
  return dict((k, d1.get(k, no) + d2.get(k, no)) for k in keys)

def generateVCRProfilesByRadius(RNG, count:int,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> List[Profile]:
    resultProfiles = {k:list() for k in radiusParams.keys()}

    for i in range(count):
        profiles = generateVCRProfileByRadius(RNG=R, C=C, V=V, radiusParams=radiusParams)
        resultProfiles = mergeDictLists(resultProfiles, profiles)

    return resultProfiles

def runnerVCRProfilesByRadius(C:int, V:int):
    RNG=default_rng()
    distribution = 'uniform'
    count = 10
    radiusParams={0:(0.7, 0.7), 1:(1.2, 1.2), 2:(0.7,1.2), 3:(0,2.5)}

    path = "resources/random/numpy/vcr-{}-{}R-{}C{}V.npy"

    profilesByR = generateVCRProfilesByRadius(RNG, count, C, V, radiusParams)

    for rParam, profiles in profilesByR.items():
        saveLoc = path.format(distribution, rParam, C, V)
        print("Saving to : {}".format(saveLoc))
        with open(saveLoc, 'wb') as f:
            np.save(file=f, arr=profiles, allow_pickle=False)


#%%
runnerVCRProfilesByRadius(30,30)


#%%
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))
d = generateVCRProfilesByRadius(RNG=R, C=20, V=20, radiusParams={0:(1,1), 1:(0,4)})
sns.heatmap((d[1][0].A), cmap=['black', 'gray'], ax=ax1)
sns.heatmap((getVCRProfileInCRVROrder(d[1][0]).A), cmap=['black', 'gray'], ax=ax2)
ax1.set_title("Initial Approval Matrix")
ax1.set_xlabel("candidates")
ax1.set_ylabel("voters")

ax2.set_title("VCR Ordered Approval Matrix")
ax2.set_xlabel("candidates (reindexed)")
ax2.set_ylabel("voters (reindexed)")

#%%
d0 = generateVCRProfileByRadius(RNG=R, C=4, V=4, radiusParams={0:(1,1), 1:(0,4)})
d1 = generateVCRProfileByRadius(RNG=R, C=4, V=4, radiusParams={0:(1,1), 1:(0,4)})


#%%
d = generateVCRProfilesByRadius(RNG=R, count=10, C=20, V=20, radiusParams={0:(1,1), 1:(0,4)})

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
P44_0 = np.load('resources/random/numpy/vcr-uniform-0R-30C30V.npy')
P44_1 = np.load('resources/random/numpy/vcr-uniform-1R-30C30V.npy')
P44_2 = np.load('resources/random/numpy/vcr-uniform-2R-30C30V.npy')
P44_3 = np.load('resources/random/numpy/vcr-uniform-3R-30C30V.npy')

#%%
print(Profile.fromNumpy(P44_3[0]))

#%%
i = 2
fig, (ax1, ax2) = plt.subplots(2,2, figsize=(14,12))
sns.heatmap(getVCRProfileInCRVROrder(Profile.fromNumpy(P44_0[i])).A, cmap=['black', 'gray'], ax=ax1[0])
sns.heatmap(getVCRProfileInCRVROrder(Profile.fromNumpy(P44_1[i])).A, cmap=['black', 'gray'], ax=ax1[1])
sns.heatmap(getVCRProfileInCRVROrder(Profile.fromNumpy(P44_2[i])).A, cmap=['black', 'gray'], ax=ax2[0])
sns.heatmap(getVCRProfileInCRVROrder(Profile.fromNumpy(P44_3[i])).A, cmap=['black', 'gray'], ax=ax2[1])

