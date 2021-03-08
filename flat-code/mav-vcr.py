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
from mavUtils import minimax, minimaxCR3, analyze, mavScore, committeeTupleToVector
from utils import shuffleRows, shuffleCols

#%%
def VCRNCOP_44():
    return np.load("resources/output/4C4V/NCOP-profiles/ncop-44-0.npy")

#%%
P = list(map(Profile.fromNumpy, VCRNCOP_44()))

#%%
analyze(profile=P[9].A, k=2, d=2, excl=False)

#%% 
i = 5
print(P[i])
V, C = P[i].V, P[i].C
V.sort(key=lambda t3: t3[1] - t3[2])
C.sort(key=lambda t3: t3[1] - t3[2])


#%%
T = shuffleRows(P[i].A, [0,3,2,1])
T = shuffleCols(T, [3,1,2,0])
T

#%%
T = shuffleRows(P[5].A, [0,2,1,3])
T = shuffleCols(T, [2,0,3,1])
T

#%%
minimax(T, k=2)

#%%
status, K = minimaxCR3(T, k=2, d=0)
score = mavScore(T, committeeTupleToVector(K, 4))
print(status, K, score)


#%%
mavScore(T, committeeTupleToVector([0,1], 4))