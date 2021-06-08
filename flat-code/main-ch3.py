#%%
from definitions import Profile, Candidate, Voter
from static_profiles import VCRNCOP_1010, VCR_dist_r_cv

from copDetectionSat import solve_sat # ROW COP --> Voter Range
from vcrDetectionAlt import detectVCRProperty, detectVRProperty, createGPEnv
from mavUtils import getVCRProfileInCRVROrder


#%%
import pandas as pd
import numpy as np
from time import time
from typing import List, Tuple, NamedTuple
import os

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

#%%
def prefHeatmap(profile):
    plt.figure(figsize=(7,5))
    sns.heatmap((getVCRProfileInCRVROrder(profile)).A, cmap=['black', 'gray'])
    plt.show()

def getVCLists(A:np.ndarray) -> Tuple[List[str],List[str]]:
    V = ['v' + str(i) for i in range(A.shape[0])]
    C = ['c' + str(i) for i in range(A.shape[1])]
    return V,C

#%%
gEnv = createGPEnv()

#%%
P1515 = VCR_dist_r_cv(15, 15, "2gauss", 4)
P4040 = VCR_dist_r_cv(40, 40, "uniformgauss", 5)
P8080 = VCR_dist_r_cv(80, 80, "uniformgauss", 5)
P100100 = VCR_dist_r_cv(100, 100, "uniformgauss", 5)


#%%
i = 1
Ps = P100100
Vs, Cs = getVCLists(Ps[i].A)

startTime = time()
satRes = solve_sat(np.copy(Ps[i].A))
print(time() - startTime)

startTime = time()
ilpRes = detectVRProperty(Ps[i].A, Cs, Vs, gEnv)
print(time() - startTime)

print("ILP = {}, SAT = {}".format(ilpRes, satRes.status))

#%%
107/0.07