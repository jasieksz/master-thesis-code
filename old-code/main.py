#%%
import numpy as np
from profiles import VCR44, notVCR33, VCR22
from definitions import Voter, Candidate, Profile, isVCR
from helpers import consecutiveOnes2D, getAllSquareProfiles,getProfilePermutations
from detectionILP import detectorMockDist, detectorPosNeg
from pprint import pprint
import math
import time
import multiprocessing
import magic
import pickle
import itertools

#%%
def parallelDetection(filePath: str, mapFun, mapIter, detectorFilter:bool) -> list:
    startTime = time.time()
    with multiprocessing.Pool(16) as p:
        res = list(p.map(mapFun, mapIter))  
    endTime = time.time()
    print(endTime - startTime)
    vcrRes = [p for (vcr,p) in res if vcr == detectorFilter]

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
def stepsPre(c, v, vcrDetector):
    basePath = '../profiles/profiles-{}C-{}V/'.format(c,v)
    allPath = 'all-P{}{}.npy'.format(c,v)
    vcrPath = 'VCR-P{}{}'.format(c,v)
    vcrFNCOPPath = 'VCR-NCOP-P{}{}'.format(c,v)

    allP = parallelProfileGeneration(c,v)
    np.save(basePath+allPath, allP)
    print(allP.shape)

    vcrP = parallelDetection(basePath+vcrPath, vcrDetector, allP, True)
    print(len(vcrP))

    vcrFullNCOP_P = parallelDetection(basePath+vcrFNCOPPath, magic.COPFilter, vcrP, False)
    print(len(vcrFullNCOP_P))

#%%
def stepsPost(c, v):
    basePath = '../profiles/profiles-{}C-{}V/'.format(c,v)
    allPath = 'all-P{}{}.npy'.format(c,v)
    vcrPath = 'VCR-P{}{}'.format(c,v)
    vcrFNCOPPath = 'VCR-NCOP-P{}{}'.format(c,v)

    allP44 = np.load(basePath+allPath)
    print(allP44.shape)

    with open(basePath+vcrPath, 'rb') as f:
        vcrP44 = pickle.load(f)
    print(len(vcrP44))
        
    with open(basePath+vcrFNCOPPath, 'rb') as f:
        vcrFullNCOP_P44 = pickle.load(f)
    print(len(vcrFullNCOP_P44))




#%%
notVcrP = parallelDetection(basePath+vcrPath, vcrDetector, allP, True)

#%%
stepsPost(4,4)