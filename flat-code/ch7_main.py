#%%
from definitions import Profile, Candidate, Voter
from vcrDetectionAlt import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv
from vcrDetectionAlt import findCRPoints, findVRPoints
from mavUtils import getVCRProfileInCROrder, getVCRProfileInVROrder, getVCRProfileInCRVROrder

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import combinations, product
from time import time
from typing import List, Tuple, NamedTuple
from functools import partial
import os


#%%
# delete columns
def deleteCandidates(A:np.ndarray, candidates:List[int]) -> np.ndarray:
    return np.delete(A, obj=candidates, axis=1)

# delete rows
def deleteVoters(A:np.ndarray, voters:List[int]) -> np.ndarray:
    return np.delete(A, obj=voters, axis=0)

def getVCLists(A:np.ndarray) -> Tuple[List[str],List[str]]:
    V = ['v' + str(i) for i in range(A.shape[0])]
    C = ['c' + str(i) for i in range(A.shape[1])]
    return V,C

def detectCRPropertyWrapper(gEnv, A, C, V):
    return detectCRProperty(A, C, V, gEnv)

def detectVRPropertyWrapper(gEnv, A, C, V):
    return detectVRProperty(A, C, V, gEnv)

def detectFCOP(A, C, V, gEnv):
    return detectVRProperty(A, C, V, gEnv) and detectCRProperty(A, C, V, gEnv)

def detectCOP(A, C, V, gEnv):
    return detectVRProperty(A, C, V, gEnv) or detectCRProperty(A, C, V, gEnv)

##################################################################
##################################################################
##################################################################

class multiDeletionSearchResults(NamedTuple):
    status:int
    kC:int
    kV:int
    combination:List[int]

class fullMultiDeletionSearch(NamedTuple):
    property:str
    axis:int
    kC:int
    kV:int

def multiCombinationDeletionSearch(A:np.ndarray, deleteAxis:int, propertyDetectionFun, gEnv) -> multiDeletionSearchResults:
    k = 1
    results = []
    sT = time()
    for dC,dV in sorted(list(product(range(1,5), range(1,5))), key=lambda t2: (t2[0] + t2[1])):
        for combC,combV in product(combinations(range(A.shape[1]), dC), combinations(range(A.shape[0]),dV)):
            tmpA = deleteCandidates(A, combC)
            tmpA = deleteVoters(tmpA, combV)
            tmpV,tmpC = getVCLists(tmpA)
            propertyStatus = propertyDetectionFun(tmpA, tmpC, tmpV, gEnv)
            if propertyStatus:
                return multiDeletionSearchResults(status=1, kC=dC, kV=dV, combination=[(combC, combV)])
    print("NOT FOUND ", time() - sT)
    return multiDeletionSearchResults(status=0, kC=dC+1, kV=dV+1, combination=[])

def fullProfileMultiSearch(profiles):
    env = createGPEnv()
    detectionFuns = {"cr":detectCRProperty, "vr":detectVRProperty}
    deleteAxes = {"cand":1, "voter":0}
    results = []
    for profile in profiles:
        for keyDel,deletionAxis in deleteAxes.items():
            minKCOP = (1000,1000)
            for keyProp,detectF in detectionFuns.items():
                res = multiCombinationDeletionSearch(np.copy(profile.A), deletionAxis, detectF, env)
                minKCOP = (res.kC, res.kV) if sum(minKCOP) > res.kC + res.kV else minKCOP
                res2 = fullMultiDeletionSearch(property=keyProp, axis=keyDel, kC=res.kC, kV=res.kV)
                results.append(res2)
            results.append(fullMultiDeletionSearch(property="cop", axis=keyDel, kC=minKCOP[0], kV=minKCOP[1]))
    return results


##################################################################
##################################################################
##################################################################

class deletionSearchResults(NamedTuple):
    status:int
    k:int
    combination:List[int]

class fullDeletionSearch(NamedTuple):
    property:str
    axis:int
    k:int


def combinationDeletionSearch(A:np.ndarray, deleteAxis:int, propertyDetectionFun, gEnv) -> deletionSearchResults:
    deleteFunction = deleteCandidates if deleteAxis == 1 else deleteVoters
    k = 1
    results = []
    while k < 5:
        sT = time()
        for combination in map(list, combinations(range(A.shape[deleteAxis]), k)):
            tmpA = deleteFunction(A, combination)
            tmpV,tmpC = getVCLists(tmpA)
            propertyStatus = propertyDetectionFun(tmpA, tmpC, tmpV, gEnv)
            if propertyStatus:
                return deletionSearchResults(1, k, combination)

        k += 1
    print("NOT FOUND ", time() - sT)
    return deletionSearchResults(0, k, [])

def fullProfileSearch(profiles):
    env = createGPEnv()
    detectionFuns = {"cr":detectCRProperty, "vr":detectVRProperty}
    deleteAxes = {"cand":1, "voter":0}
    results = []
    for profile in profiles:
        for keyDel,deletionAxis in deleteAxes.items():
            minKCOP = 20
            for keyProp,detectF in detectionFuns.items():
                res = combinationDeletionSearch(np.copy(profile.A), deletionAxis, detectF, env)
                minKCOP = min(minKCOP, res.k)
                res2 = fullDeletionSearch(property=keyProp, axis=keyDel, k=res.k)
                results.append(res2)
            results.append(fullDeletionSearch(property="cop", axis=keyDel, k=minKCOP))
    return results
    
def NCOP_1010():
    return list(map(Profile.fromNumpy, np.load("resources/random/numpy/ncop/ncop-2gauss-4R-10C10V.npy")))

def NCOP_2020():
    return list(map(Profile.fromNumpy, np.load("resources/random/numpy/ncop/ncop-2gauss-4R-20C20V.npy")))

def NCOP_1212():
    return list(map(Profile.fromNumpy, np.load("resources/random/numpy/ncop/ncop-2gauss-4R-12C12V.npy")))

def NCOP_CV(C:int, V:int):
    return list(map(Profile.fromNumpy, np.load("resources/random/numpy/ncop/ncop-2gauss-4R-{}C{}V.npy".format(C,V))))

def prefHeatmap(profile):
    plt.figure(figsize=(5,4))
    sns.heatmap((getVCRProfileInCRVROrder(profile)).A, cmap=['black', 'gray'])
    plt.show()

#%%
#############################################
#############################################
# NOTEBOOK

P1515 = NCOP_CV(15,15)
len(P1515)

#%%
env = createGPEnv()


#%%
start = time()
res2 = fullProfileSearch(P1515[70:72])
print("TIME : ", time() - start)

#%%
res2

#%%
dfAll = pd.concat([dfAll, pd.DataFrame(res2)])
dfAll.describe()
dfAll2 = dfAll
dfAll2['deleted agent'] = dfAll2['axis']

#%%
df = pd.read_csv("resources/random/ch7-transform/ncop-2gauss-4R-15C15V-0S-200E.csv")
df['deleted agent'] = df['axis']
df.describe()

#%%
g = sns.violinplot(data=df, x='property', y='k', hue='deleted agent', split=True, inner="stick", scale="area")
g.set_title("Deleting k agents to transform TVCR into VR / CR\n 200 Profiles with 15 Candidates 15 Voters")

#%%
start = time()
res3 = fullProfileMultiSearch(P1515[10:11])
print("TIME : ", time() - start)
res3

#%%
dfMulti = pd.concat([dfMulti, pd.DataFrame(res3)])
dfMulti["k"] = dfMulti['kC'] + dfMulti['kV']
dfMulti.describe()

#%%
sns.violinplot(data=dfMulti, x='property', y='k', hue='axis', split=True, inner="stick")

#%%
basePath = "resources/random/ch7-transform"
paths = [e for e in os.listdir(basePath) if e[-3:] == "csv"]
df = pd.concat((pd.read_csv("{}/{}".format(basePath, filePath)) for filePath in paths))
df['deleted agent'] = df['axis']
df.describe()

#%%
g = sns.violinplot(data=df, x='property', y='k', hue='deleted agent', split=True, inner="stick", scale="area")
g.set_title("Deleting k agents to transform TVCR into VR / CR\n 1000 Profiles with 15 Candidates 15 Voters\nR=2Gauss, X=2Gauss")


#%%
df.groupby(['k', 'property', 'axis']).count().reset_index()


#%%
### GRID VIOLIN VIS
############
############
############
D = ['2gauss', 'uniform', 'gaussuniform', 'uniformgauss']
R = [4,8]
paths = ["resources/random/ch7-transform/{}-{}R/{}".format(dist,r,e) for dist in D for r in R  for e in os.listdir("resources/random/ch7-transform/{}-{}R".format(dist, r)) if e[-3:] == "csv"]

#%%
radiusParams={
    4:"G(1.5,0.5) G(1.5,0.5)",
    5:"U(0.7,1.2) G(1.5,0.5)",
    6:"G(1.5,0.5) U(0.7,1.2)",
    7:"U(0.7,1.2) U(0.7,1.2)",
    8:"U(0,3) U(0,3)"
}

distParams={
    'uniform': "Uniform Cands\n Uniform Voters",
    '2gauss': "Gauss Cands\n Gauss Voters",
    'uniformgauss': "Uniform Cands\n Gauss Voters",
    'gaussuniform': "Gauss Cands\n Uniform Voters"
}

#%%
transformDf = pd.concat((pd.read_csv(path) for path in paths))
transformDf['deleted agent'] = transformDf['axis']
transformDf['R'] = transformDf['R'].map(radiusParams)
transformDf['distribution'] = transformDf['distribution'].map(distParams)
transformDf.describe()

#%%
tDfUniform = transformDf.loc[(transformDf['distribution'] == 'uniform')]
tDfGauss = transformDf.loc[(transformDf['distribution'] == '2gauss')]
tDf1 = pd.concat([tDfUniform, tDfGauss])

#%%
sns.catplot(kind='violin',
    data=transformDf, x='distribution', y='k',
    hue='deleted agent', col='property', row='R',
    split=True, sharex=False, sharey=True)

savePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter4/Figs/15-15-tr-grid-x{}-col{}-row{}.png"
plt.savefig(savePath.format('dist', 'prop', 'R'))

#%%
print(transformDf.head(), sep=',')

#%%
def combs():
    res = []
    for dC,dV in sorted(list(product(range(1,5), range(1,5))), key=lambda t2: (t2[0] + t2[1])):
        for combC,combV in product(combinations(range(7), dC), combinations(range(7),dV)):
            res.append((combC, combV))
    return res
