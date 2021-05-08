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

def profileFromVotersAndCandidates(voters: np.ndarray, candidates: np.ndarray) -> Profile:
    A = np.zeros((len(voters),len(candidates)))
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

def generateVCRProfilesByRadius(RNG,
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
        profile = profileFromVotersAndCandidates(voters, candidates)
        resultProfiles[key].append(profile)
    
    return resultProfiles

#%%
d = generateVCRProfilesByRadius(RNG=R, C=20, V=20, radiusParams={0:(1,1), 1:(0,4)})
sns.heatmap((getVCRProfileInCRVROrder(d[0][0]).A), cmap=['black', 'gray'])
plt.show()
sns.heatmap((getVCRProfileInCRVROrder(d[1][0]).A), cmap=['black', 'gray'])

