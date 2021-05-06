#%%
from vcrDetectionAlt import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv
from vcrDomain import isVCR, vcrPropertyRaw
from definitions import Profile

import numpy as np
from numpy.random import default_rng
import pandas as pd
import seaborn as sns

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

    voters = generateDoubleGaussRandomAgents(RNG=RNG, count=V, proportion=proportionV,
        meanMajor=meanMajor, meanMinor=meanMinor, std=std, radiusConst=radiusConst)

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
    candidates = agentRandomFunction(RNG=RNG, count=C, rMin=0, rMax=40, xMin=0, xMax=100)
    voters = agentRandomFunction(RNG, V, 0, 20, 0, 100)

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

P = generateRandomVCRProfile(R, 20, 20, generateUniformRandomAgents)

sns.heatmap(P.A, cmap=['black', 'gray'])

vIds, cIds = getVCLists(P.A) 
cr = detectCRProperty(P.A, cIds, vIds, gEnv)
vr = detectVRProperty(P.A, cIds, vIds, gEnv)

print(cr, vr)

#%%


#%%
df = pd.DataFrame([v.x for v in P.V], columns=['x'])
sns.displot(data=df, x='x')

#%%
