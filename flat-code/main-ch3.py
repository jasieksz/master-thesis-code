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
time()

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
    status:bool
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
                results.append(IlpSatResult(time=res.time, status=res.status, algo=algoName, electionSize=eName, profileI=i, runNumber=runNumber))
    return results

#%%
# res = runner(gEnv, 2)

df = pd.DataFrame(res)
df['Election Size'] = df['electionSize'].map({
    "small":"15 Candidates\n15Voters",
    "medium":"20 Candidates\n20Voters",
    "large":"40 Candidates\n40Voters",
})
df['Algorithm'] = df['algo']
df['Detection time'] = df['time']
df['Detection status'] = df['status']
df.head()

#%%
# dfAll.to_csv('resources/random/ch3-ilp-sat.csv', header=True, index=False)

#%%
dfAll = pd.concat([dfAll, df])
dfAll.head()

#%%
dfAll = pd.read_csv('resources/random/ch3-ilp-sat.csv')
df2 = dfAll.groupby(['Algorithm', 'Election Size', 'Detection status', 'profileI', 'status']).min()
df2 = df2.reset_index()
df2['Election Size'] = df2['Election Size'].map({
    "15 Candidates\n15Voters": "15 Candidates 15 Voters",
    "20 Candidates\n20Voters": "20 Candidates 20 Voters",
    "40 Candidates\n40Voters": "40 Candidates 40 Voters",
})
df2.head()

#%%
dfStat = df2.groupby(['Algorithm', 'Election Size'])[['Algorithm', 'Election Size', 'time']].describe(percentiles=[])
dfStat
#%%
dfStat = df2.groupby(['Algorithm', 'Election Size'])[['Algorithm', 'Election Size', 'time']].describe(percentiles=[])
dfStat.columns = dfStat.columns.droplevel()

# tableSavePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter3/Figs/table-ex.tex"
# dfStat[['mean', 'std',]] \
#     .to_latex(buf=tableSavePath,
#         float_format="%.3f",
#         caption="Statistical metrics for detection time, gropued by algorithm and election size.",
#         multirow=True
#     )

#%%
for k,v in {
    "15 Candidates\n15Voters": "15 Candidates 15 Voters",
    "20 Candidates\n20Voters": "20 Candidates 20 Voters",
    "40 Candidates\n40Voters": "40 Candidates 40 Voters",
}.items():
    g = sns.catplot(kind="violin", data=df2[df2['Election Size'] == v], x="Detection time", y="Algorithm", col="Election Size",
        hue="Detection status", split=True,
        sharex=False, orient="h", inner="point", scale="count", palette=["dodgerblue", "lightskyblue"],
        height=12, aspect=1.2)

    g.set_titles("{col_name}", size=14)
    g.set_xlabels("Detection time, seconds", size=14)
    g.set_ylabels("Algorithm", size=14)
    g.set_yticklabels(["ILP", "SAT"], size=12)

    if v != "40 Candidates 40 Voters":
        g._legend.remove()
    g.tight_layout()

    # savePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter3/Figs/ex1-violin-{}.png".format(v[:2])
    # plt.savefig(savePath)

#%%
blues4 = cm.get_cmap('Blues_r', 10)
palette=[blues4(3), blues4(7)]


sns.set(font_scale = 1.1)
g = sns.catplot(kind="violin", data=df2, x="Detection time", y="Algorithm", col="Election Size",
    hue="Detection status", split=True,
    sharex=False, orient="h", inner="point", scale="count",
    palette=[blues4(2), blues4(6)],
    height=3.75, aspect=1.1, legend=False)

g.set_titles("{col_name}", size=14)
g.set_xlabels("Detection time, seconds", size=14)
g.set_ylabels("Algorithm", size=14)
g.set_yticklabels(["ILP", "SAT"], size=14)
plt.legend(bbox_to_anchor=(0.5, 0.35), loc=2, borderaxespad=0., title="Detection status")
g.tight_layout()

# savePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter3/Figs/alt-ex1-violin.png"
# plt.savefig(savePath)

#%%
g = sns.catplot(kind="box", data=dfAll,
    x="Detection time", y="Algorithm", col="Election Size", hue="Detection status",
    sharex=False, orient="h", palette=["dodgerblue", "lightskyblue"])

g.tight_layout()
# savePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter3/Figs/ex1-box.png"
# plt.savefig(savePath)

#%%
g = sns.catplot(kind="swarm", data=df2, x="Detection time", y="Algorithm",
    hue="Detection status", col="Election Size", sharex=False, sharey=False,
    orient="h", palette=["dodgerblue", "lightskyblue"])

g.tight_layout()
# savePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter3/Figs/ex1-swarm.png"
# plt.savefig(savePath)

#%%
