# %%
import numpy as np
from numpy.random import default_rng
from vcrDomain import isVCR, vcrPropertyRaw
from definitions import Profile
import sys
from time import time

#%%
def generateRandomAgents(RNG, count:int,
        rMin:int, rMax:int,
        xMin:int, xMax:int) -> np.ndarray:
    positions = RNG.uniform(low=xMin, high=xMax, size=count)
    radii = RNG.uniform(low=rMin, high=rMax, size=count)
    return np.dstack((positions, radii))[0]

def generateRandomVCRProfile(RNG, C:int, V:int) -> Profile:
    candidates = generateRandomAgents(RNG, C, 0, 100, 0, 750)
    voters = generateRandomAgents(RNG, V, 0, 100, 0, 750)

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
    if len(sys.argv) != 4:
        print("args: C, V, subSet")
    else:    
        C = int(sys.argv[1])
        V = int(sys.argv[2])
        subSet = int(sys.argv[3])

        path = "resources/input/{}C{}V/VCR-{}.npy".format(C, V, subSet)

        RNG = default_rng()
        profiles = [generateRandomVCRProfile(RNG, C, V) for i in range(100000)]
        if not all(map(isVCR, profiles)):
            print("BOOM")
        else:
            npProfiles = np.array(list(map(Profile.asNumpy, profiles)))
            with open(path, 'wb') as f:
                np.save(file=f, arr=npProfiles, allow_pickle=False)
    print(time() - startTime)
            
#%%
# path = "resources/input/{}C{}V/VCR-{}.npy".format(4,4,0)
# profiles = np.load(path)

# #%%
# P = Profile.fromNumpy(profiles[5])
# print(P)
# isVCR(P)