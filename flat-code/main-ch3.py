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
from functools import partial
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
P200200 = VCR_dist_r_cv(200, 200, "uniformgauss", 5)


#%%
i = 124
Ps =  P4040
Vs, Cs = getVCLists(Ps[i].A)
satRes = solve_sat(np.copy(Ps[i].A))
ilpRes = detectVRProperty(Ps[i].A, Cs, Vs, gEnv)

print("ILP = {}\nSAT = {}".format(ilpRes, satRes))

#%%
class IlpSatResult(NamedTuple):
    time:float
    algo:str
    electionSize:str
    profileI:int
    runNumber:int

def wrapperSAT(A:np.ndarray):
    return solve_sat(A)

def wrapperILP(gEnv, A:np.ndarray):
    Vs, Cs = getVCLists(A)
    return detectVRProperty(A, Cs, Vs, gEnv)

#%%
def runner(gEnv, runNumber:int):
    algo = {"ILP":partial(wrapperILP, gEnv), "SAT":partial(wrapperSAT)}
    electionSize = {"small":(15,15), "medium":(20,20), "large":(40,40)}
    results = []
    for eName, eSize in electionSize.items():
        approvalMatrices = [profile.A for profile in VCR_dist_r_cv(eSize[0], eSize[1], "uniformgauss", 4)[:100]]
        for i,approvalMatrix in enumerate(approvalMatrices):
            print(i)
            for algoName, algoFun in algo.items():
                res = algoFun(approvalMatrix)
                results.append(IlpSatResult(time=res.time, algo=algoName, electionSize=eName, profileI=i, runNumber=runNumber))
    return results

#%%
res = runner(gEnv, 0)

#%%
df = pd.DataFrame(res)
df['Election Size'] = df['electionSize'].map({
    "small":"15 Candidates\n15Voters",
    "medium":"20 Candidates\n20Voters",
    "large":"40 Candidates\n40Voters",
})
df['Algorithm'] = df['algo'].map({"ilp":"ILP", "sat":"SAT"})
df['Detection time'] = df['time']

#%%
df.head()

#%%
g = sns.catplot(kind="violin", data=df, x="Detection time", y="Algorithm", col="Election Size",
    sharex=False, orient="h", inner="point", scale="count", palette=["dodgerblue", "lightskyblue"])

g.tight_layout()
savePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter3/Figs/ex1-violin.png"
plt.savefig(savePath)

#%%
g = sns.catplot(kind="box", data=df, x="Detection time", y="Algorithm", col="Election Size",
    sharex=False, orient="h", palette=["dodgerblue", "lightskyblue"])

g.tight_layout()
savePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter3/Figs/ex1-box.png"
plt.savefig(savePath)

#%%
g = sns.catplot(kind="swarm", data=df, x="Detection time", y="Algorithm",
    col="Election Size", sharex=False, sharey=False,
    orient="h", palette=["dodgerblue", "lightskyblue"])

g.tight_layout()
savePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter3/Figs/ex1-swarm.png"
plt.savefig(savePath)

#%%
