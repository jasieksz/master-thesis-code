#%%
import numpy as np
from definitions import Profile, Candidate, Voter, VCR44
from itertools import combinations
from typing import List, Tuple, Dict
import math
from numpy import ndarray
from vcrDetectionAlt import findCRPoints
from collections import namedtuple
from pprint import pprint
from time import time
from mavUtils import minimax, minimaxCR3, analyze, mavScore, committeeTupleToVector, basePartialCompare
from utils import shuffleRows, shuffleCols
from functools import partial

#%%
def VCRNCOP_44():
    return np.load("resources/output/4C4V/NCOP-profiles/ncop-44-0.npy")

def VCRNCOP_55():
    return np.load("resources/output/5C5V/NCOP-profiles/ncop-55-1.npy")


def VCRNCOP_55_2():
    return np.load("resources/output/5C5V/NCOP-profiles/ncop-55-2.npy")

def bruteMAVWrapper(k:int, A:np.ndarray):
    return minimax(A=A, k=k)

def crMAVWrapper(k:int, d:int, A:np.ndarray):
    return minimaxCR3(A=A, k=k, d=d)

def singleCompareMAVs(A, k, d):
    bruteMAVPartial = partial(bruteMAVWrapper, k)
    crMAVPartial = partial(crMAVWrapper, k, d)
    return basePartialCompare(A=A, algoX=bruteMAVPartial, algoY=crMAVPartial)

def compareMAVs(profiles, k, d):
    results = [singleCompareMAVs(profile.A, k, d) for profile in profiles]
    falseResults = {i:result for i,result in enumerate(results) if not result.status}
    return falseResults

#%%
P44 = list(map(Profile.fromNumpy, VCRNCOP_44()))
P55 = list(map(Profile.fromNumpy, VCRNCOP_55_2()))

#%%
compareMAVs(profiles=P55, k=3, d=5).keys()

#%%
crMAVWrapper(k=3, d=4, A=P55[9].A)

#%%
print(P55[13])