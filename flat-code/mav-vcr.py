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
from functools import partial, reduce

#%%
def VCRNCOP_44():
    return np.load("resources/output/4C4V/NCOP-profiles/ncop-44-0.npy")

def VCRNCOP_55():
    return np.load("resources/output/5C5V/NCOP-profiles/ncop-55-1.npy")

def VCRNCOP_55_2():
    return np.load("resources/output/5C5V/NCOP-profiles/ncop-55-2.npy")

def VCRNCOP_66():
    return np.load("resources/output/6C6V/NCOP-profiles/ncop-66-0.npy")

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

def parameterCompareMAVs(profiles, k, dRange):
    S = [set(compareMAVs(profiles=profiles, k=k, d=d).keys()) for d in dRange]
    return S[0].intersection(*S)

#%%
P44 = list(map(Profile.fromNumpy, VCRNCOP_44()))
P55 = list(map(Profile.fromNumpy, VCRNCOP_55()))
P55_2 = list(map(Profile.fromNumpy, VCRNCOP_55_2()))
P66 = list(map(Profile.fromNumpy, VCRNCOP_66()))

#%%
parameterCompareMAVs(profiles=P66, k=2, dRange=range(7))

#%%
singleCompareMAVs(P55_2[13].A, k=3, d=5)

#%%
analyze(profile=P66[27].A, k=2, d=7)