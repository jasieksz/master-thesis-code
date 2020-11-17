#%%
import numpy as np
from profiles import VCR44, notVCR33, VCR22
from definitions import Voter, Candidate, Profile, isVCR
from helpers import consecutiveOnes2D, getAllSquareProfiles
from detectionILP import detectorMockDist, detectorPosNeg
from pprint import pprint
import math
import time
import multiprocessing
import magic
import pickle
import itertools

#%%
P44 = np.load('../profiles/P44.npy')
P44.shape

#%%
def parallelVCRDetection(filePath: str, mapFun, mapIter) -> list:
    startTime = time.time()
    with multiprocessing.Pool(16) as p:
        res = list(p.map(mapFun, mapIter))  
    endTime = time.time()
    print(endTime - startTime)
    vcrRes = [p for (vcr,p) in res if vcr]

    with open(filePath, 'wb') as f:
        pickle.dump(vcrRes, f)

    return vcrRes

#%%
def parallelProfileGeneration(candidate, voter):
    startTime = time.time()
    with multiprocessing.Pool(1) as p:
        singleVotes = list(p.map(list, itertools.product([0,1], repeat=candidate)))
        profiles = list(p.map(list, itertools.product(singleVotes, repeat=voter)))
        npProfiles = list(p.map(magic.npArray, profiles))
    endTime = time.time()
    print(endTime - startTime)
    return np.array(npProfiles)


#%%
parallelProfileGeneration(4,4).shape


