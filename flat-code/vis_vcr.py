#%%
from functools import partial

import numpy as np
from numpy import ndarray

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib import ticker
sns.set_style("darkgrid", {"axes.facecolor": ".8"})

from typing import NamedTuple, List, Dict
from definitions import Profile


#%%
voteCounter = lambda arr,column: np.sum() # List[int], indices of 1s in a given column

class Agent(NamedTuple):
    id:str
    start:float
    end:float
    offset:float
    voteCount:int
    color:str

def vcrProfileToAgents(profile:Profile) -> List[Agent]:
    oB = 0.25
    agents = []
    voteCounts = sum(profile.A)
    for y,c,vC in zip(np.arange(oB, len(profile.C)/2 + oB, oB), profile.C, voteCounts):
        a = Agent(id=c.id,
                start=c.x - c.r,
                end=c.x + c.r,
                offset=y,
                voteCount=vC,
                color='dimgray')
        agents.append(a)
    for y,v in zip(np.arange(-oB, -len(profile.V)/2 - oB, -oB), profile.V):
        a = Agent(id=v.id,
                start=v.x - v.r,
                end=v.x + v.r,
                offset=y,
                voteCount=0,
                color='slategrey')
        agents.append(a)
    return agents

def vcrProfileToAgentsWithDeletion(profile:Profile, deleteC:List, deleteV:List):
    agents = vcrProfileToAgents(profile)
    result = []
    for a in agents:
        if a.id in deleteC or a.id in deleteV:
            tmpA = Agent(a.id, a.start, a.end, a.offset, 'green')
            result.append(tmpA)
        else:
            result.append(a)
    return result

def vcrProfileToAgentsWithColors(profile:Profile, colors:Dict) -> List[Agent]:
    agents = vcrProfileToAgents(profile)
    result = []
    for a in agents:
        tmpA = Agent(a.id, a.start, a.end, a.offset, a.voteCount, colors.get(a.id, a.color))
        result.append(tmpA)
    return result
   

def formatter(agents, y, pos):
    return [a.id for a in agents if a.offset == y][0]

def plotVCRAgents(agents:List[Agent]) -> None:

    viridis = cm.get_cmap('viridis', len(agents)) 
    oB = 0.075
    df = pd.DataFrame(agents)
    plt.figure(figsize=(10,8))
    for i,a in enumerate(agents):
        p1 = plt.hlines(y=a.offset,
                    xmin=a.start,
                    xmax=a.end,
                    label=a.id,
                    color=a.color,#viridis(i/len(agents)),
                    linewidth=4)

        p2 = plt.vlines(x=a.start,
                        ymin=a.offset-oB,
                        ymax=a.offset+oB, 
                        color=a.color,#viridis(i/len(agents)),
                        linewidth=4)

        p3 = plt.vlines(x=a.end,
                        ymin=a.offset-oB,
                        ymax=a.offset+oB, 
                        color=a.color,#viridis(i/len(agents)),
                        linewidth=4)

    plt.xlim([df.start.min()-0.5, df.end.max()+0.5])
    plt.ylim([df.offset.min()-0.5, df.offset.max()+0.5])
    plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper center')
    plt.grid(b=True, axis='x')
    plt.yticks([a.offset for a in agents], [a.id + " vc=" + str(a.voteCount) for a in agents])
    partialFormatter = partial(formatter, agents)
    # plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(partialFormatter))
    plt.show()

#%%
def run():
    plotVCRAgents(vcrProfileToAgents(P66[10]))
    plotVCRAgents(vcrProfileToAgentsWithDeletion(P66[99], ['C0'], ['V0','V2']))


#%%
df = pd.read_csv('resources/random/spark/30C30V/stats-merged.csv')
df

#%%
sns.catplot(data=df, x='distribution', y='count',
    hue='property', col='R', col_wrap=2,
    orient="v", kind='bar', sharex=False)

