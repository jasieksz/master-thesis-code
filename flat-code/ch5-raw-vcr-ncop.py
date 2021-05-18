#%%
from vcrDetectionAlt import detectVCRProperty, detectCRProperty, detectVRProperty, createGPEnv
from definitions import Profile

#%%
import sys
import numpy as np
import pandas as pd
from time import time
from typing import List, Tuple, Dict

#%%
def partitionPropertyMapper(profiles, candidatesIds, votersIds):
    env = createGPEnv()
    for profile in profiles:
        crResult = detectCRProperty(A=profile.A, C=candidatesIds, V=votersIds, env=env)
        vrResult = detectVRProperty(A=profile.A, C=candidatesIds, V=votersIds, env=env)
        status = 3
        if (crResult and not vrResult):
            status = 1
        elif (not crResult and vrResult):
            status = 2
        elif (not crResult and not vrResult):
            status = 0
        yield (status, profile.asNumpy())

def runner(start:int, end:int):
    C = 40
    V = 40
    R = 4
    distribution = 'gaussuniform'

    propertyType = "vcr"
    baseInPath = "resources/random/numpy/{}-{}-{}R-{}C{}V.npy".format(propertyType, distribution, R, C, V)
    baseOutProfilesPath = "resources/random/numpy/{}-{}-{}R-{}C{}V-{}S-{}E.npy".format("ncop", distribution, R, C, V, start, end)
    baseOutStatsPath = "resources/random/pandas/{}-{}-{}R-{}C{}V-{}S-{}E.csv".format("ncop", distribution, R, C, V, start, end)


    print("\nLoading from : {}\nSaving to : {}\n".format(baseInPath, baseOutProfilesPath))

    propertyStatus = {0:"ncop", 1:"cr", 2:"vr", 3:"cop"}
    statistics = {}
    candidatesIds = ['C' + str(i) for i in range(C)]
    votersIds = ['V' + str(i) for i in range(V)]

    profiles = map(Profile.fromNumpy, np.load(baseInPath)[start:end])

    profileStats = list(partitionPropertyMapper(profiles, candidatesIds, votersIds))

    ncopProfiles = np.array(list(map(lambda t2: t2[1], filter(lambda t2: t2[0] == 0, profileStats))))

    with open(baseOutProfilesPath, 'wb') as f:
        np.save(file=f, arr=ncopProfiles, allow_pickle=False)


    stats = pd.DataFrame(map(lambda t2: t2[0], profileStats), columns=['property'])
    aggStats = pd.DataFrame({propertyStatus[k[0]]:v for k,v in stats.value_counts().to_dict().items()}.items(), columns=['property', 'count'])
    aggStats['distribution'] = distribution
    aggStats['R'] = R
    aggStats.to_csv(baseOutStatsPath, index=False, header=True)
    return aggStats

#%%
if __name__ == "__main__":
    s = int(sys.argv[1])
    e = int(sys.argv[2])
    sT = time()
    runner(s, e)
    print(time() - sT)
