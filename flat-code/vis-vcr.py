#%%
from functools import partial

import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import cm

from typing import NamedTuple, List
from definitions import Profile


#%%
def VCRNCOP_44():
    return np.load("resources/output/4C4V/NCOP-profiles/ncop-44-0.npy")

def VCRNCOP_55_1():
    return np.load("resources/output/5C5V/NCOP-profiles/ncop-55-1.npy")

def VCRNCOP_55_2():
    return np.load("resources/output/5C5V/NCOP-profiles/ncop-55-2.npy")

def VCRNCOP_55_3():
    return np.load("resources/output/5C5V/NCOP-profiles/ncop-55-3.npy")

def VCRNCOP_66():
    return np.load("resources/output/6C6V/NCOP-profiles/ncop-66-0.npy")

#%%
P44 = list(map(Profile.fromNumpy, VCRNCOP_44()))
P66 = list(map(Profile.fromNumpy, VCRNCOP_66()))

#%%
class Agent(NamedTuple):
    id:str
    start:float
    end:float
    offset:float
    color:str

def vcrProfileToAgents(profile:Profile) -> List[Agent]:
    oB = 0.25
    agents = []
    for y,c in zip(np.arange(oB, len(profile.C)/2 + oB, oB), profile.C):
        a = Agent(id=c.id,
                start=c.x - c.r,
                end=c.x + c.r,
                offset=y,
                color='red')
        agents.append(a)
    for y,v in zip(np.arange(-oB, -len(profile.V)/2 - oB, -oB), profile.V):
        a = Agent(id=v.id,
                start=v.x - v.r,
                end=v.x + v.r,
                offset=y,
                color='blue')
        agents.append(a)
    return agents

def plotVCRAgents(agents:List[Agent]) -> None:
    viridis = cm.get_cmap('seismic', len(agents)) 
    oB = 0.05
    df = pd.DataFrame(agents)
    plt.figure(figsize=(10,8))
    for i,a in enumerate(agents):
        p1 = plt.hlines(y=a.offset,
                    xmin=a.start,
                    xmax=a.end,
                    label=a.id,
                    color=viridis(i/len(agents)))

        p2 = plt.vlines(x=a.start,
                        ymin=a.offset-oB,
                        ymax=a.offset+oB, 
                        color=viridis(i/len(agents)))

        p3 = plt.vlines(x=a.end,
                        ymin=a.offset-oB,
                        ymax=a.offset+oB, 
                        color=viridis(i/len(agents)))

    plt.xlim([df.start.min()-1, df.end.max()+1])
    plt.ylim([df.offset.min()-1, df.offset.max()+1])
    plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper center')
    plt.grid(b=True, axis='x')
    plt.show()

#%%
plotVCRAgents(vcrProfileToAgents(P66[10]))
