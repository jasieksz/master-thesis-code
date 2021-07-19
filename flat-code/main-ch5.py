#%%
from vcrDetectionAlt import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv, findVRPoints, findCRPoints
from vcrDomain import isVCR, vcrPropertyRaw
from definitions import Profile
from mavUtils import getVCRProfileInCRVROrder,getVCROrders,getVCRProfileInVROrder,getVCRProfileInCROrder
from matplotlib import pyplot as plt

#%%
import os
import numpy as np
from numpy.random import default_rng
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from time import time
from typing import List, Tuple, Dict, NamedTuple
from functools import partial


#%%
# HELPERS
##################################################
##################################################
def getVCLists(A:np.ndarray):
    V = ['v' + str(i) for i in range(A.shape[0])]
    C = ['c' + str(i) for i in range(A.shape[1])]
    return V,C

def mergeDictLists(d1, d2):
  keys = set(d1).union(d2)
  no = []
  return dict((k, d1.get(k, no) + d2.get(k, no)) for k in keys)

def npProfileFromVotersAndCandidates(voters: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    A = np.zeros((len(voters),len(candidates)))
    for vI, (vX,vR) in enumerate(voters):
        for cI, (cX,cR) in enumerate(candidates):
            if vcrPropertyRaw(vX, vR, cX, cR):
                A[vI,cI] = 1
    
    return np.concatenate([
            np.array(A.shape),
            candidates.flatten(),
            voters.flatten(),
            A.flatten()])    


def uniformRadiusWrapper(RNG, low, high, size):
    return RNG.uniform(low=low, high=high, size=size)

def gaussRadiusWrapper(RNG, mean, std, size):
    return RNG.normal(mean, std, size)

def normalizeRadius(radii:np.ndarray) -> np.ndarray:
    m = np.min(radii)
    if m < 0:
        radii -= m
    return radii


#%%
# UNIFORM
##################################################
##################################################

def generateVCRProfileByRadiusUniform(RNG,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> Profile:

    resultProfiles = {k:list() for k in radiusParams.keys()}
    xMin = -10
    xMax = 10

    for key, (rCFun,rVFun) in radiusParams.items():

        cPositions = RNG.uniform(low=xMin, high=xMax, size=C)
        vPositions = RNG.uniform(low=xMin, high=xMax, size=V)
        radiiC = normalizeRadius(rCFun(C))
        radiiV = normalizeRadius(rVFun(V))

        candidates = np.dstack((cPositions, radiiC))[0]
        voters = np.dstack((vPositions, radiiV))[0]
        profile = npProfileFromVotersAndCandidates(voters, candidates)
        resultProfiles[key].append(profile)
    
    return resultProfiles

def generateVCRProfilesByRadiusUniform(RNG, count:int,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> List[Profile]:
    resultProfiles = {k:list() for k in radiusParams.keys()}

    for i in range(count):
        profiles = generateVCRProfileByRadiusUniform(RNG=R, C=C, V=V, radiusParams=radiusParams)
        resultProfiles = mergeDictLists(resultProfiles, profiles)

    return resultProfiles

def runnerVCRProfilesByRadiusUniform(C:int, V:int):
    RNG=default_rng()
    distribution = 'uniform'
    count = 100

    radiusParams={
        4:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        5:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        6:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        7:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        8:(partial(uniformRadiusWrapper, RNG, 0, 3), partial(uniformRadiusWrapper, RNG, 0, 3)),
    }

    path = "resources/random/numpy/vcr-{}-{}R-{}C{}V.npy"

    profilesByR = generateVCRProfilesByRadiusUniform(RNG, count, C, V, radiusParams)

    for rParam, profiles in profilesByR.items():
        saveLoc = path.format(distribution, rParam, C, V)
        print("Saving to : {}".format(saveLoc))
        with open(saveLoc, 'wb') as f:
            np.save(file=f, arr=profiles, allow_pickle=False)


#%%
# 2 GAUSS
##################################################
##################################################
def generateVCRProfileByRadius2Gauss(RNG,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> Profile:

    resultProfiles = {k:list() for k in radiusParams.keys()}

    majorityC = int(C * 0.7)
    minorityC = C - majorityC
    majorityV = int(V * 0.7)
    minorityV = V - majorityV

    positionsCMajor = RNG.normal(-3, 1.8, size=majorityC)
    positionsCMinor = RNG.normal(3, 1.8, size=minorityC)
    cPositions = np.append(positionsCMajor, positionsCMinor)

    positionsVMajor = RNG.normal(-3, 1.8, size=majorityV)
    positionsVMinor = RNG.normal(3, 1.8, size=minorityV)
    vPositions = np.append(positionsVMajor, positionsVMinor)

    for key, (rCFun,rVFun) in radiusParams.items():
        radiiC = normalizeRadius(rCFun(C))
        radiiV = normalizeRadius(rVFun(V))



        candidates = np.dstack((cPositions, radiiC))[0]
        voters = np.dstack((vPositions, radiiV))[0]
        profile = npProfileFromVotersAndCandidates(voters, candidates)
        resultProfiles[key].append(profile)
    
    return resultProfiles

def generateVCRProfilesByRadius2Gauss(RNG, count:int,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> List[Profile]:
    resultProfiles = {k:list() for k in radiusParams.keys()}

    for i in range(count):
        profiles = generateVCRProfileByRadius2Gauss(RNG=R, C=C, V=V, radiusParams=radiusParams)
        resultProfiles = mergeDictLists(resultProfiles, profiles)

    return resultProfiles

def runnerVCRProfilesByRadius2Gauss(C:int, V:int):
    RNG=default_rng()
    distribution = '2gauss'
    count = 100

    radiusParams={
        4:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        5:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        6:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        7:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        8:(partial(uniformRadiusWrapper, RNG, 0, 3), partial(uniformRadiusWrapper, RNG, 0, 3)),
    }

    path = "resources/random/numpy/vcr-{}-{}R-{}C{}V.npy"

    profilesByR = generateVCRProfilesByRadius2Gauss(RNG, count, C, V, radiusParams)

    for rParam, profiles in profilesByR.items():
        saveLoc = path.format(distribution, rParam, C, V)
        print("Saving to : {}".format(saveLoc))
        with open(saveLoc, 'wb') as f:
            np.save(file=f, arr=profiles, allow_pickle=False)

#%%
# Gauss Uniform
##################################################
##################################################
def generateVCRProfileByRadiusGaussUniform(RNG,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> Profile:

    resultProfiles = {k:list() for k in radiusParams.keys()}

    majorityC = int(C * 0.7)
    minorityC = C - majorityC

    positionsCMajor = RNG.normal(-3, 1.8, size=majorityC)
    positionsCMinor = RNG.normal(3, 1.8, size=minorityC)
    cPositions = np.append(positionsCMajor, positionsCMinor)
    vPositions = RNG.uniform(low=-10, high=10, size=V)

    for key, (rCFun,rVFun) in radiusParams.items():
        radiiC = normalizeRadius(rCFun(C))
        radiiV = normalizeRadius(rVFun(V))


        
        candidates = np.dstack((cPositions, radiiC))[0]
        voters = np.dstack((vPositions, radiiV))[0]
        profile = npProfileFromVotersAndCandidates(voters, candidates)
        resultProfiles[key].append(profile)
    
    return resultProfiles

def generateVCRProfilesByRadiusGaussUniform(RNG, count:int,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> List[Profile]:
    resultProfiles = {k:list() for k in radiusParams.keys()}

    for i in range(count):
        profiles = generateVCRProfileByRadiusGaussUniform(RNG=R, C=C, V=V, radiusParams=radiusParams)
        resultProfiles = mergeDictLists(resultProfiles, profiles)

    return resultProfiles

def runnerVCRProfilesByRadiusGaussUniform(C:int, V:int):
    RNG=default_rng()
    distribution = 'gaussuniform'
    count = 4000
    radiusParams={
        4:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        # 5:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        # 6:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        # 7:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        8:(partial(uniformRadiusWrapper, RNG, 0, 3), partial(uniformRadiusWrapper, RNG, 0, 3)),
    }

    path = "resources/random/numpy/vcr-{}-{}R-{}C{}V.npy"

    profilesByR = generateVCRProfilesByRadiusGaussUniform(RNG, count, C, V, radiusParams)

    for rParam, profiles in profilesByR.items():
        saveLoc = path.format(distribution, rParam, C, V)
        print("Saving to : {}".format(saveLoc))
        with open(saveLoc, 'wb') as f:
            np.save(file=f, arr=profiles, allow_pickle=False)

#%%
# Uniform Gauss
##################################################
##################################################
def generateVCRProfileByRadiusUniformGauss(RNG,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> Profile:

    resultProfiles = {k:list() for k in radiusParams.keys()}

    majorityV = int(C * 0.7)
    minorityV = V - majorityV

    for key, (rCFun,rVFun) in radiusParams.items():

        cPositions = RNG.uniform(low=-10, high=10, size=V)
        positionsVMajor = RNG.normal(-3, 1.8, size=majorityV)
        positionsVMinor = RNG.normal(3, 1.8, size=minorityV)
        vPositions = np.append(positionsVMajor, positionsVMinor)

        radiiC = normalizeRadius(rCFun(C))
        radiiV = normalizeRadius(rVFun(V))
        
        candidates = np.dstack((cPositions, radiiC))[0]
        voters = np.dstack((vPositions, radiiV))[0]
        profile = npProfileFromVotersAndCandidates(voters, candidates)
        resultProfiles[key].append(profile)
    
    return resultProfiles

def generateVCRProfilesByRadiusUniformGauss(RNG, count:int,
    C:int, V:int,
    radiusParams: Dict[int,Tuple[float,float]]) -> List[Profile]:
    resultProfiles = {k:list() for k in radiusParams.keys()}

    for i in range(count):
        profiles = generateVCRProfileByRadiusUniformGauss(RNG=R, C=C, V=V, radiusParams=radiusParams)
        resultProfiles = mergeDictLists(resultProfiles, profiles)

    return resultProfiles

def runnerVCRProfilesByRadiusUniformGauss(C:int, V:int):
    RNG=default_rng()
    distribution = 'uniformgauss'
    count = 100
    radiusParams={
        # 4:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        5:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(gaussRadiusWrapper, RNG, 1.5, 0.5)),
        # 6:(partial(gaussRadiusWrapper, RNG, 1.5, 0.5), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        # 7:(partial(uniformRadiusWrapper, RNG, 0.7, 1.2), partial(uniformRadiusWrapper, RNG, 0.7, 1.2)),
        # 8:(partial(uniformRadiusWrapper, RNG, 0, 3), partial(uniformRadiusWrapper, RNG, 0, 3)),
    }
    path = "resources/random/numpy/vcr-{}-{}R-{}C{}V.npy"

    profilesByR = generateVCRProfilesByRadiusUniformGauss(RNG, count, C, V, radiusParams)

    for rParam, profiles in profilesByR.items():
        saveLoc = path.format(distribution, rParam, C, V)
        print("Saving to : {}".format(saveLoc))
        with open(saveLoc, 'wb') as f:
            np.save(file=f, arr=profiles, allow_pickle=False)

##################################################
# NOTEBOOK
##################################################

#%%
R = default_rng()
gEnv = createGPEnv()

# #%%
# runnerVCRProfilesByRadiusUniform(8,8)

# #%%
# runnerVCRProfilesByRadius2Gauss(15,15)

# #%%
# runnerVCRProfilesByRadiusGaussUniform(15,15)

# #%%
# runnerVCRProfilesByRadiusUniformGauss(300,300)


#%%

#%%
positionNames = {
    "uniform" :         "UCP/UVP",
    "2gauss" :          "GCP/GVP",
    "uniformgauss" :    "UCP/GVP",
    "gaussuniform" :    "GCP/UVP",
}

radiiNames = {
    4 : "GCR/GVR",
    5 : "SUCR/GVR",
    6 : "GCR/SUVR",
    7 : "SUCR/SUVR",
    8 : "LUCR/LUVR",
}


def f(C,V):
    for dist in ['uniform','2gauss','uniformgauss','gaussuniform']:
        for r in range(4,9):
            path = [e for e in os.listdir("resources/random/spark/{}C{}V/ncop-{}-{}R-stats/".format(C, V, dist, r)) if e[-3:] == "csv"][0]
            df = pd.read_csv("resources/random/spark/{}C{}V/ncop-{}-{}R-stats/{}".format(C, V, dist, r, path))
            dist2 = dist
            if dist == 'gaussuniform':
                dist2 = 'gauss C\n uniform V'
            if dist == 'uniformgauss':
                dist2 = 'uniform C\n gauss V'
            df['distribution'] = dist2
            df['R'] = radiusParams[r]
            yield df

def f2():
    for dist in ['uniformgauss','gaussuniform']:
        for r in range(5,8):
            print(mergePandasStats(dist, r, False))

def mergePandasStats(distribution, R, write=False):
    paths = [e for e in os.listdir("resources/random/pandas/{}-{}R".format(distribution, R)) if e[-3:] == "csv"]
    df = pd.concat((pd.read_csv("resources/random/pandas/{}-{}R/{}".format(distribution, R, path)) for path in paths))
    df2 = df.groupby(['property', 'distribution', 'R']).sum().reset_index()
    df2 = df2[['property', 'count', 'distribution', 'R']]
    if write:
        df2.to_csv("resources/random/pandas/{}-{}R-merged.csv".format(distribution, R), header=True, index=False)
    return df2

def fullMergePandasStats4040(write=False):
    basePath = "resources/random/pandas"
    paths = [e for e in os.listdir(basePath) if e[-3:] == "csv"]
    df = pd.concat((pd.read_csv("{}/{}".format(basePath, filePath)) for filePath in paths))
    if not write:
        df['R'] = df['R'].map(radiiNames)
        df['distribution'] = df['distribution'].map(positionNames)
        df['property'] = df['property'].map({"ncop":"TVCR", "vr":"VR", "cr":"CR", "cop":"FCOP"})
    if write:
        df.to_csv("resources/random/merged-40C40V-stats.csv", header=True, index=False)
    return df

def fullMergePandasStats2020(write=False):
    basePath = "resources/random/pandas-20C20V"
    paths = [e for e in os.listdir(basePath) if e[-3:] == "csv"]
    df = pd.concat((pd.read_csv("{}/{}".format(basePath, filePath)) for filePath in paths))
    if not write:
        df['R'] = df['R'].map(radiiNames)
        df['distribution'] = df['distribution'].map(positionNames)
        df['property'] = df['property'].map({"ncop":"TVCR", "vr":"VR", "cr":"CR", "cop":"FCOP"})
    if write:
        df.to_csv("resources/random/merged-20C20V-stats.csv", header=True, index=False)
    return df


#%%
### EX.2 AGGREGATED STATS
###############
###############
df4040 = fullMergePandasStats4040(False)
df2020 = fullMergePandasStats2020(False)


#%%
catOrder = {
    "uniform" :         "UCP/UVP",
    "2gauss" :          "GCP/GVP",
    "uniformgauss" :    "UCP/GVP",
    "gaussuniform" :    "GCP/UVP",
}

colOrder = {
    4 : "GCR/GVR",
    5 : "SUCR/GVR",
    6 : "GCR/SUVR",
    7 : "SUCR/SUVR",
    8 : "LUCR/LUVR",
}

for k,v in colOrder.items():
    g = sns.catplot(data=df2020[df2020['R'] == v], x='distribution', y='count',
        hue='property',
        orient="v", kind='bar', sharex=True, sharey=True, 
        order=catOrder.values(), hue_order=["TVCR", "CR", "VR", "FCOP"],
        palette="Blues_r",#["dodgerblue", "lightskyblue", "powderblue", "royalblue"],
        legend_out=True, height=4, aspect=1,)
    
    plt.grid(b=True, axis='y')

    if True or k != 5:
        g._legend.remove()

    # g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("{}".format(v), y=0.93, x=0.45, size=14)
    g.set(xlabel="", ylabel="")
    if k == 7 or k == 6:
        g.set(ylabel="Count")
    g.set(yticks=[0, 200, 400, 600, 800, 1000], ylim=(0,1000))

    # g.set_yticklabels(g.get_yticklabels(), size=12)
    for axes in g.axes.flat:
        _ = axes.set_xticklabels(axes.get_xticklabels(), size=11)
        axes.set_yticklabels([0, 200, 400, 600, 800, 1000], size=11)
        axes.set_ylabel(axes.get_ylabel(), size=12)

    # g.yticks(size=15)
    # g.fig.subplots_adjust(top=0.9)
    g.tight_layout()
    savePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter4/Figs/ex2-2020-stats-R{}.png".format(v.replace("/", "-"))
    plt.savefig(savePath, bbox_inches='tight')

#%%
#### EX.1 SMALL PROFILES
#####################
####################

#%%
exOneDf = pd.read_csv("resources/output/merged-stats-small-vcr.csv")
totals = exOneDf.groupby(['election'])['count'].sum().to_dict()
exOneDf.reset_index()
exOneDf['count'] = exOneDf.apply(lambda row: row['count'] / totals[row['election']], axis=1)
exOneDf['property'] = exOneDf['property'].map({"vcr":"VCR", "ncop":"TVCR","not vcr":"NOT VCR"})
exOneDf.head()


#%%
plt.figure(figsize=(7,4))


g = sns.barplot(data=exOneDf,
    x='election', y='count', hue='property',
    palette= sns.color_palette("Blues_r", 4),#["dodgerblue", "lightskyblue", "royalblue"],
    orient="v", log=False,
    order=['3C3V', '4C4V', '4C6V', '6C4V', '5C5V'])

plt.grid(b=True, axis='y')

g.set(yticklabels=["0%", "20%", "40%", "60%", "80%", "100%"], ylim=(0,1),
    xticklabels=['{} Candidates\n{} Voters'.format(c,v) for c,v in zip([3,4,4,6,5],[3,4,6,4,5])])

g.set_title("How many profiles are TVCR and VCR", size=14)
g.set_ylabel("Percentage of all profiles", size=14)
g.set_xlabel("Election Size", size=14)

g.spines['right'].set_visible(False)
g.spines['top'].set_visible(False)

plt.legend(bbox_to_anchor=(1.01,1.025))
plt.tight_layout()
# savePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter4/Figs/ex1-small-stats.png"
# plt.savefig(savePath)


#%%
#### HEATMAP GRID
#####################
####################
class heatmapGridProfile(NamedTuple):
    R:str
    dist:str
    A:np.ndarray


C = 20
V = 20
R = default_rng()

radiusParams={
    4:(partial(gaussRadiusWrapper, R, 1.5, 0.5), partial(gaussRadiusWrapper, R, 1.5, 0.5)),
    5:(partial(uniformRadiusWrapper, R, 0.7, 1.2), partial(gaussRadiusWrapper, R, 1.5, 0.5)),
    6:(partial(gaussRadiusWrapper, R, 1.5, 0.5), partial(uniformRadiusWrapper, R, 0.7, 1.2)),
    7:(partial(uniformRadiusWrapper, R, 0.7, 1.2), partial(uniformRadiusWrapper, R, 0.7, 1.2)),
    8:(partial(uniformRadiusWrapper, R, 0, 3), partial(uniformRadiusWrapper, R, 0, 3)),
}

distFuns = {
    'uniform':generateVCRProfileByRadiusUniform,
    '2gauss':generateVCRProfileByRadius2Gauss, 
    'gaussuniform':generateVCRProfileByRadiusGaussUniform, 
    'uniformgauss':generateVCRProfileByRadiusUniformGauss
}

# radiusNames={
#     4:"Radii Model\nG(1.5,0.5) G(1.5,0.5)",
#     5:"Radii Model\nU(0.7,1.2) G(1.5,0.5)",
#     6:"Radii Model\nG(1.5,0.5) U(0.7,1.2)",
#     7:"Radii Model\nU(0.7,1.2) U(0.7,1.2)",
#     8:"Radii Model\nU(0,3) U(0,3)"
# }

# distNames={
#     'uniform':      "Position Model\nUniform Cands Uniform Voters",
#     '2gauss':       "Position Model\nGauss Cands Gauss Voters",
#     'uniformgauss': "Position Model\nUniform Cands Gauss Voters",
#     'gaussuniform': "Position Model\nGauss Cands Uniform Voters"
# }


#%%
profiles = []
for dName, dFun in distFuns.items():
    profilesByR = dFun(R, C, V, radiusParams)
    for rName,profile in profilesByR.items():
        pA = getVCRProfileInCRVROrder(Profile.fromNumpy(profile[0])).A
        profiles.append(heatmapGridProfile(rName, dName, pA))

hmDf = pd.DataFrame(profiles)
# hmDf['R'] = hmDf['R'].map(radiusNames)
# hmDf['X'] = hmDf['dist'].map(distNames)

hmDf.head()

#%%
positionNames = {
    "uniform" :         "UCP/UVP",
    "2gauss" :          "GCP/GVP",
    "uniformgauss" :    "UCP/GVP",
    "gaussuniform" :    "GCP/UVP",
}

radiiNames = {
    8 : "LUCR/LUVR",
    4 : "GCR/GVR",
    7 : "SUCR/SUVR",
    5 : "SUCR/GVR",
    6 : "GCR/SUVR",
}

def hm(data, color):
    g = sns.heatmap(data=list(data.head(1)['A'])[0],
        cmap=["white", "dodgerblue"],

        cbar=False,
        square=True,
        linewidths=1,
        linecolor="dodgerblue")

g = sns.FacetGrid(data=hmDf, row='dist', col='R',
    col_order=radiiNames, row_order=positionNames,
    margin_titles=False,
    sharex=True, sharey=True)
g.map_dataframe(hm)
g.set_titles("")


# g.fig.suptitle("Approval Matrices", x=0.5, y=1)

for ax,col in zip(g.axes[0], radiiNames.values()):
    ax.set_title(col, size=20)

for ax,row in zip(g.axes[:,0], positionNames.values()):
    ax.set_ylabel(row, size=20)

g.tight_layout()

savePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter4/Figs/{}-{}-examples-grid-blue.png".format(C,V)
plt.savefig(savePath)

#%%
# sns.color_palette("colorblind")

#%%
from static_profiles import VCR_dist_r_cv

#%%
ncopProfiles88 = VCR_dist_r_cv(c=8, v=8, dist="uniform", r=4)


#%%
P = ncopProfiles88[27]


#%%
oV, oC = getVCROrders(P)
print(oV,oC)

#%%
blues = cm.get_cmap('Blues', 4)
reds = cm.get_cmap('Reds_r', 30) 

#%%
g = sns.heatmap(data=np.array([[0,1,0,1],[0,1,1,0],[1,1,0,0],[1,1,1,1]]),
        cmap=['white', 'dodgerblue'],
        cbar=False,
        square=True,
        linewidths=1,
        linecolor='dodgerblue')

bottom, top = g.get_ylim()
g.set_ylim(bottom + 0.5, top - 0.5)

left, right = g.get_xlim()
g.set_xlim(left - 0.5, right + 0.5)

g.set_xticklabels(["$c_{}$".format(i) for i in range(4)], size=15)
g.set_yticklabels(["$v_{}$".format(i)  for i in range(4)], rotation=0, size=15)
# savePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter2/Figs/ncop-election-example-matrix-ordered.png"
savePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter2/Figs/alt-ncop-election-example-matrix-unordered.png"
# plt.savefig(savePath)

#%%
vrP = getVCRProfileInVROrder(P)
vs,cs = getVCLists(vrP.A)
status,vrRes = findVRPoints(vrP.A, cs, vs, gEnv)
print(status)
vrP = Profile.fromILPRes(vrP.A, vrRes, cs, vs)
# def fromILPRes(approvalSet: ndarray, result, candidatesIds: List[str], votersIds: List[str]):
plotVCRAgents(vcrProfileToAgents(vrP))

#%%
crP = getVCRProfileInCROrder(P)
vs,cs = getVCLists(crP.A)
status,crRes = findCRPoints(crP.A, cs, vs, gEnv)
print(status)
crP = Profile.fromILPRes(crP.A, crRes, cs, vs)
# def fromILPRes(approvalSet: ndarray, result, candidatesIds: List[str], votersIds: List[str]):
#%%
plotVCRAgents(vcrProfileToAgents(ncopP))


#%%
crP.V


#%%
# FIG 2.1
blues = cm.get_cmap('Blues_r', 30)
reds = cm.get_cmap('Reds_r', 20) 

a = [
    Agent(id='$c_{0}$', start=-1.8351453636138408, end=1.7506167491466733, offset=0.1, voteCount=5.0, color=blues(4)),
    Agent(id='$c_{1}$', start=-11.527826688856981, end=-8.203615190316137, offset=0.1, voteCount=1.0, color=blues(10)),
    Agent(id='$c_{2}$', start=-1.2067156473537903, end=1.5723029235328891, offset=0.25, voteCount=5.0, color=blues(5)),
    Agent(id='$c_{3}$', start=4.915115112886295, end=8.01783520320441, offset=0.1, voteCount=2.0, color=blues(8)),
    Agent(id='$c_{4}$', start=2.220205219483197, end=4.280813902864915, offset=0.1, voteCount=3.0, color=blues(6)),
    Agent(id='$c_{5}$', start=-6.023265177151032, end=-1.9645963904199455, offset=0.25, voteCount=2.0, color=blues(7)),
    Agent(id='$c_{6}$', start=5.7806088173421815, end=9.528232683008707, offset=0.25, voteCount=2.0, color=blues(8)),
    Agent(id='$c_{7}$', start=-6.468132439916875, end=-4.933741380390366, offset=0.1, voteCount=1.0, color=blues(9)),

    Agent(id='$v_{0}$', start=-2.3759867449911707, end=2.210519911524838, offset=-0.1, voteCount=0, color=reds(4)),
    Agent(id='$v_{1}$', start=2.555029419570911, end=8.387086548828673, offset=-0.1, voteCount=0, color=reds(5)),
    Agent(id='$v_{2}$', start=-9.195449043662265, end=-6.4547813600306005, offset=-0.1, voteCount=0, color=reds(10)),
    Agent(id='$v_{3}$', start=-2.29929435640923, end=0.5163411205005624, offset=-0.2, voteCount=0, color=reds(6)),
    Agent(id='$v_{4}$', start=1.4864059713818079, end=3.8433228938193253, offset=-0.2, voteCount=0, color=reds(7)),
    Agent(id='$v_{5}$', start=0.5366110728767322, end=4.331297846813388, offset=-0.3, voteCount=0, color=reds(8)),
    Agent(id='$v_{6}$', start=7.324233401045559, end=9.586242307133405, offset=-0.2, voteCount=0, color=reds(11)),
    Agent(id='$v_{7}$', start=-1.9028841560294307, end=-0.15220580276382978, offset=-0.3, voteCount=0, color=reds(9))
    ]

plotVCRAgents(a)


#%%
a = [Agent(id='P', start=9, end=12, offset=0.2, voteCount=5.0, color='dodgerblue'),
 Agent(id='C1', start=0, end=3, offset=0.1, voteCount=1.0, color='dodgerblue'),
 Agent(id='C2', start=2.8, end=6, offset=0.2, voteCount=5.0, color='dodgerblue'),
 Agent(id='C3', start=6.4, end=8.6, offset=0.1, voteCount=2.0, color='dodgerblue'),
 Agent(id='C4', start=11.5, end=16, offset=0.1, voteCount=3.0, color='dodgerblue'),

 Agent(id='V0', start=-0.5, end=1.5, offset=-0.1, voteCount=0, color='lightskyblue'),
 Agent(id='V1', start=2, end=3.5, offset=-0.15, voteCount=0, color='lightskyblue'),
 Agent(id='V2', start=3.5, end=4.5, offset=-0.1, voteCount=0, color='lightskyblue'),
 Agent(id='V3', start=4.2, end=5.75, offset=-0.17, voteCount=0, color='lightskyblue'),
 Agent(id='V4', start=4.5, end=8, offset=-0.24, voteCount=0, color='lightskyblue'),
 Agent(id='V5', start=7.2, end=8.4, offset=-0.1, voteCount=0, color='lightskyblue'),
 Agent(id='V6', start=8, end=11, offset=-0.15, voteCount=0, color='lightskyblue'),
 Agent(id='V7', start=9.5, end=11, offset=-0.25, voteCount=0, color='lightskyblue'),
 Agent(id='V8', start=11, end=13, offset=-0.19, voteCount=0, color='lightskyblue'),
 Agent(id='V9', start=13.5, end=15, offset=-0.1, voteCount=0, color='lightskyblue'),
 Agent(id='V10', start=14, end=17, offset=-0.17, voteCount=0, color='lightskyblue')]

plotVCRAgents(a)

#%%
# FIG 2.2B
vra = [

    # Agent(id='C0', start=0.0, end=0.0, offset=0.1, voteCount=1.0, color='dodgerblue'),
    # Agent(id='C1', start=0.0, end=0.0, offset=0.2, voteCount=1.0, color='dodgerblue'),
    # Agent(id='C2', start=1.0, end=1.0, offset=0.3, voteCount=2.0, color='dodgerblue'),
    # Agent(id='C3', start=2.0, end=2.0, offset=0.4, voteCount=5.0, color='dodgerblue'),
    # Agent(id='C4', start=2.0, end=2.0, offset=0.5, voteCount=5.0, color='dodgerblue'),
    # Agent(id='C5', start=3.0, end=3.0, offset=0.6, voteCount=3.0, color='dodgerblue'),
    # Agent(id='C6', start=4.0, end=4.0, offset=0.7, voteCount=2.0, color='dodgerblue'),
    # Agent(id='C7', start=4.0, end=4.0, offset=0.8, voteCount=2.0, color='dodgerblue'),

    Agent(id='$v_{0}$', start=1.0, end=2.0, offset=-0.05, voteCount=0, color=reds(7)),
    Agent(id='$v_{1}$', start=3.0, end=4.0, offset=-0.05, voteCount=0, color=reds(10)),
    Agent(id='$v_{2}$', start=0.0, end=0.0, offset=-0.05, voteCount=0, color=reds(5)),
    Agent(id='$v_{3}$', start=1.0, end=2.0, offset=-0.00, voteCount=0, color=reds(6)),
    Agent(id='$v_{4}$', start=2.0, end=3.0, offset=-0.15, voteCount=0, color=reds(8)),
    Agent(id='$v_{5}$', start=2.0, end=3.0, offset=-0.10, voteCount=0, color=reds(9)),
    Agent(id='$v_{6}$', start=4.0, end=4.0, offset=-0.15, voteCount=0, color=reds(11)),
    Agent(id='$v_{7}$', start=2.0, end=2.0, offset=-0.22, voteCount=0, color=reds(10))]

plotVCRAgents(vra)


#%%
# FIG 2.2A
cra = [
    Agent(id='$c_{0}$', start=0.0, end=2.0, offset=0.1, voteCount=5.0, color=blues(6)),
    Agent(id='$c_{1}$', start=5.0, end=5.0, offset=0.1, voteCount=1.0, color=blues(9)),
    Agent(id='$c_{2}$', start=0.0, end=2.0, offset=0.2, voteCount=5.0, color=blues(4)),
    Agent(id='$c_{3}$', start=3.0, end=4.0, offset=0.1, voteCount=2.0, color=blues(5)),
    Agent(id='$c_{4}$', start=2.0, end=3.0, offset=0.15, voteCount=3.0, color=blues(7)),
    Agent(id='$c_{5}$', start=1.0, end=1.0, offset=0.3, voteCount=2.0, color=blues(10)),
    Agent(id='$c_{6}$', start=3.0, end=4.0, offset=0.2, voteCount=2.0, color=blues(8)),
    Agent(id='$c_{7}$', start=5.0, end=5.0, offset=0.2, voteCount=1.0, color=blues(11)),
    # Agent(id='V0', start=5.0, end=5.0, offset=-0.1, voteCount=0, color='lightskyblue'),
    # Agent(id='V1', start=0.0, end=0.0, offset=-0.2, voteCount=0, color='lightskyblue'),
    # Agent(id='V2', start=1.0, end=1.0, offset=-0.30000000000000004, voteCount=0, color='lightskyblue'),
    # Agent(id='V3', start=1.0, end=1.0, offset=-0.4, voteCount=0, color='lightskyblue'),
    # Agent(id='V4', start=2.0, end=2.0, offset=-0.5, voteCount=0, color='lightskyblue'),
    # Agent(id='V5', start=2.0, end=2.0, offset=-0.6, voteCount=0, color='lightskyblue'),
    # Agent(id='V6', start=3.0, end=3.0, offset=-0.7000000000000001, voteCount=0, color='lightskyblue'),
    # Agent(id='V7', start=4.0, end=4.0, offset=-0.8, voteCount=0, color='lightskyblue')
]

plotVCRAgents(cra)

#%%
vs,cs = getVCLists(ncopP.A)
status,crRes = detectVCRProperty(ncopP.A, cs, vs, gEnv)
crRes

#%%
ncopa = [
    Agent(id='$c_{0}$', start=2.0, end=2.0, offset=0.35, voteCount=2.0, color=blues(4)),
    Agent(id='$c_{1}$', start=-2.0, end=2.0, offset=0.2, voteCount=2.0, color=blues(5)),
    Agent(id='$c_{2}$', start=0.0, end=0.0, offset=0.35, voteCount=2.0, color=blues(6)),
    Agent(id='$c_{3}$', start=1.0, end=1.0, offset=0.35, voteCount=4.0, color=blues(7)),
    Agent(id='$v_{0}$', start=1.0, end=1.0, offset=-0.05, voteCount=0, color=reds(4)),
    Agent(id='$v_{1}$', start=0.0, end=0.0, offset=-0.05, voteCount=0, color=reds(5)),
    Agent(id='$v_{2}$', start=2.0, end=2.0, offset=-0.05, voteCount=0, color=reds(6)),
    Agent(id='$v_{3}$', start=-2.0, end=2.0, offset=-0.2, voteCount=0, color=reds(7))
]
plotVCRAgents(ncopa)

#%%
# FIG 5.2
blues = cm.get_cmap('Blues_r', 40)
reds = cm.get_cmap('Reds_r', 20) 

a = [Agent(id='$p$', start=9, end=12, offset=0.15, voteCount=5.0, color='forestgreen'),
 Agent(id='$c_1$', start=0, end=3, offset=0.1, voteCount=1.0, color='black'),
 Agent(id='$c_{2}$', start=2.8, end=6, offset=0.15, voteCount=5.0, color=reds(7)),
 Agent(id='$c_{3}$', start=6.4, end=8.6, offset=0.1, voteCount=2.0, color=reds(6)),
 Agent(id='$c_{4}$', start=11.7, end=16, offset=0.1, voteCount=3.0, color=reds(5)),

 Agent(id='$v_{0}$', start=-0.5, end=1.5, offset=-0.02, voteCount=0, color=blues(4)),
 Agent(id='$v_{1}$', start=2, end=3.5, offset=-0.02, voteCount=0, color=blues(5)),
 Agent(id='$v_{2}$', start=3.5, end=4.5, offset=-0.1, voteCount=0, color=blues(6)),
 Agent(id='$v_{3}$', start=4.2, end=5.75, offset=-0.15, voteCount=0, color=blues(7)),
 Agent(id='$v_{4}$', start=4.5, end=8, offset=-0.02, voteCount=0, color=blues(8)),
 Agent(id='$v_{5}$', start=7.2, end=8.4, offset=-0.1, voteCount=0, color=blues(9)),
 Agent(id='$v_{6}$', start=8, end=11.1, offset=-0.15, voteCount=0, color='lightskyblue'),
 Agent(id='$v_{7}$', start=9.5, end=10.8, offset=-0.02, voteCount=0, color='lightskyblue'),
 Agent(id='$v_{8}$', start=11, end=13, offset=-0.1, voteCount=0, color='lightskyblue'),
 Agent(id='$v_{9}$', start=13.5, end=15, offset=-0.1, voteCount=0, color=blues(8)),
 Agent(id='$v_{10}$', start=14, end=17, offset=-0.02, voteCount=0, color=blues(9))]

plotVCRAgents(a)

#%%
# FIG 5.3
blues = cm.get_cmap('Blues_r', 40)
reds = cm.get_cmap('Reds_r', 20) 

a = [Agent(id='$p$', start=9, end=12, offset=0.05, voteCount=5.0, color='black'),
 Agent(id='$c_{E}$', start=1.5, end=8, offset=0.1, voteCount=5.0, color=reds(4)),
 Agent(id='$c_{I}$', start=2.5, end=5, offset=0.05, voteCount=2.0, color=reds(12)),

 Agent(id='$v_{1}$', start=1, end=3, offset=-0.02, voteCount=0, color=blues(21)),
 Agent(id='$v_{2}$', start=3.5, end=4.5, offset=-0.02, voteCount=0, color=blues(21)),
 Agent(id='$v_{3}$', start=3.2, end=5.5, offset=-0.06, voteCount=0, color=blues(21)),
 Agent(id='$v_{4}$', start=5.5, end=7, offset=-0.02, voteCount=0, color=blues(5)),
 Agent(id='$v_{5}$', start=5.9, end=8.8, offset=-0.06, voteCount=0, color=blues(5)),
 Agent(id='$v_{6}$', start=9, end=11.1, offset=-0.06, voteCount=0, color='black'),
 Agent(id='$v_{7}$', start=9.5, end=10.8, offset=-0.02, voteCount=0, color='black'),
 Agent(id='$v_{8}$', start=11, end=13, offset=-0.02, voteCount=0, color='black')]

plotVCRAgents(a)

#%%
# FIG 5.5
blues = cm.get_cmap('Blues_r', 30)
reds = cm.get_cmap('Reds_r', 30) 

a = [
    Agent(id='$v_{1}$', start=0, end=0, offset=0.11, voteCount=5.0, color='lightskyblue'),
    Agent(id='$v_{3}$', start=0, end=1, offset=0.05, voteCount=2.0, color='lightskyblue'),
    Agent(id='$v_{4}$', start=1, end=2, offset=0.23, voteCount=2.0, color=blues(4)),
    Agent(id='$v_{0}$', start=1, end=3, offset=0.17, voteCount=2.0, color=blues(5)),
    Agent(id='$v_{2}$', start=1, end=5, offset=0.11, voteCount=2.0, color=blues(6)),
    Agent(id='$v_{5}$', start=2, end=2, offset=0.29, voteCount=2.0, color=blues(7)),
    Agent(id='$v_{6}$', start=2, end=5, offset=0.35, voteCount=2.0, color=blues(3)),
]

plotVCRAgents(a)
