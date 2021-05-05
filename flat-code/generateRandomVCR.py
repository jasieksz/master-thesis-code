#%%
import numpy as np
from numpy.random import default_rng
from typing import Callable
import sys 
from time import time

#%%
from vcrDomain import isVCR, vcrPropertyRaw
from definitions import Profile


#%%
def generateUniformRandomAgents(RNG, count:int,
        rMin:int, rMax:int,
        xMin:int, xMax:int) -> np.ndarray:
    positions = RNG.uniform(low=xMin, high=xMax, size=count)
    radii = RNG.uniform(low=rMin, high=rMax, size=count)
    return np.dstack((positions, radii))[0]

def generateDoubleGaussRandomAgents(RNG, count:int,
        rMin:int, rMax:int,
        xMin:int, xMax:int) -> np.ndarray:
    
    majority = int(count * 0.7)
    minority = count - majority

    positionsMajor = RNG.normal(-2, 1.2, size=majority)
    positionsMinor = RNG.normal(2, 0.8, size=minority)
    radii = RNG.uniform(low=0, high=0.2, size=count)
    positions = np.append(positionsMajor, positionsMinor)
    return np.dstack((positions, radii))[0]

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
if __name__ == "__main__":    
    startTime = time()
    if len(sys.argv) != 6:
        print("args: C, V, subset, count, dist")
    else:    
        C = int(sys.argv[1])
        V = int(sys.argv[2])
        subSet = int(sys.argv[3])
        count = int(sys.argv[4])
        distribution = sys.argv[5]     

        path = "resources/random/numpy/vcr-{}-{}C{}V-{}S.npy".format(distribution, C, V, subSet)
        print("Saving to: {}".format(path))
        agentGenerator = generateUniformRandomAgents if distribution == "uniform" else generateDoubleGaussRandomAgents if distribution == "gauss" else None


        RNG = default_rng()

        profiles = [generateRandomVCRProfile(RNG, C, V, agentGenerator) for i in range(count)]
        if not all(map(isVCR, profiles)):
            print("BOOM")
        else:
            npProfiles = np.array(list(map(Profile.asNumpy, profiles)))
            with open(path, 'wb') as f:
                np.save(file=f, arr=npProfiles, allow_pickle=False)
    print(time() - startTime)
            
# #%%
# R = default_rng()

# #%%
# a = generateDoubleGaussRandomAgents(R, 100000, 0, 0, 0, 0)[:,0]
import pandas as pd
import seaborn as sns

# #%%
# R.normal(1, 0.8, size=(4))


# #%%
# generateDoubleGaussRandomAgents(R, 5, 0, 0, 0, 0)[:,0]

#%%
A = np.load("resources/random/numpy/vcr-{}-{}C{}V-{}S.npy".format("uniform", 10, 10, 1))
P = [Profile.fromNumpy(a) for a in A]

df = pd.DataFrame([np.sum(p.A) / 100 for p in P], columns=["x"])
sns.displot(data=df, x="x")


#%%
print(Profile.fromNumpy(A[90]))

#%%