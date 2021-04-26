#%%
import multiprocessing
from numpy import array, copy, ndarray, save, concatenate
from numpy.random import default_rng, SeedSequence
import numpy as np
from itertools import product, permutations
from typing import List
from functools import partial
from time import time
from vcrDomain import vcrPropertyRaw, isVCR
from matplotlib import pyplot as plt


#%%
def parallelGenerateRandomVCRPoints(cpu: int, count:int, C: int, V: int):
    seq = SeedSequence(np.random.randint(0,10000))
    random_generators = [default_rng(s) for s in seq.spawn(16)]
    gen = partial(generateRandomVCRPoints, random_generators, C, V)

    with multiprocessing.Pool(cpu) as pool:
        profilesAsNp = list(pool.map(gen, range(count)))
    return np.array(profilesAsNp)

def generateRandomVCRPoints(randomGenerators, cSize, vSize, threadIndex) -> ndarray:
    # C = np.absolute(randomGenerators[threadIndex%16].normal(loc=0.0, scale=10, size=(cSize,2)))
    # V = np.absolute(randomGenerators[threadIndex%16].normal(loc=0.0, scale=10, size=(vSize,2)))

    C = randomGenerators[threadIndex%16].integers(0, 1000, (cSize,2))
    V = randomGenerators[threadIndex%16].integers(0, 1000, (vSize,2))
    print(C)
    A = np.zeros((vSize, cSize))
    for c, (cX,cR) in enumerate(C):
        for v,(vX,vR) in enumerate(V):
            if vcrPropertyRaw(vX, vR, cX, cR):
                A[v,c] = 1
    
    
    numpyProfile = concatenate([
            array(A.shape),
            array([(cX, cR) for cX, cR in C]).flatten(),
            array([(vX, vR) for vX, vR in V]).flatten(),
            A.flatten()])  

    return numpyProfile


#%%
if __name__ == "__main__":
    path = "resources/input/20C20V/VCR2020-full-{}.npy"
    for i in range(3):
        st = time()
        VCR_P = parallelGenerateRandomVCRPoints(16, 1000, 20, 20)
        with open(path.format(i), 'wb') as f:
            save(f, VCR_P)
        print("TIME : ", time() - st)


#%%
P = parallelGenerateRandomVCRPoints(1,1,4,4)
P = Profile.fromNumpy(P[0])
print(P)
isVCR(P)

#%%
np.absolute(np.array([0,-1,1]))
