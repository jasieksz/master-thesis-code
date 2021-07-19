#%%
from functools import partial

import numpy as np
from numpy import ndarray

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib import ticker
# sns.set_style("whitegrid", {"axes.facecolor": "1"})

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
    oB = 0.1
    agents = []
    voteCounts = sum(profile.A)
    for y,c,vC in zip(np.arange(oB, len(profile.C)/2 + oB, oB), profile.C, voteCounts):
        a = Agent(id=c.id,
                start=c.x - c.r,
                end=c.x + c.r,
                offset=y,
                voteCount=vC,
                color='dodgerblue')
        agents.append(a)
    for y,v in zip(np.arange(-oB, -len(profile.V)/2 - oB, -oB), profile.V):
        a = Agent(id=v.id,
                start=v.x - v.r,
                end=v.x + v.r,
                offset=y,
                voteCount=0,
                color='lightskyblue')
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

    viridis = cm.get_cmap('Blues', len(agents)) 
    viridis2 = cm.get_cmap('Reds', len(agents)*2) 

    oB = 0.01
    df = pd.DataFrame(agents)
    fig = plt.figure(figsize=(4,3))
    for i,a in enumerate(agents):
        if True or 'V' not in a.id:
            p1 = plt.hlines(y=a.offset,
                        xmin=a.start,
                        xmax=a.end,
                        label=a.id,
                        color=a.color,#viridis(i/len(agents) + 0.6) if 'C' in a.id else viridis2(i/len(agents)),
                        linewidth=3)

            plt.text(a.start + (a.end - a.start)/2, a.offset+0.005, a.id, ha='center', va='bottom', size=14)

            p2 = plt.vlines(x=a.start,
                            ymin=a.offset-oB,
                            ymax=a.offset+oB, 
                            color=a.color,#viridis(i/len(agents) + 0.6) if 'C' in a.id else viridis2(i/len(agents)),
                            linewidth=3)

            p3 = plt.vlines(x=a.end,
                            ymin=a.offset-oB,
                            ymax=a.offset+oB, 
                            color=a.color,#viridis(i/len(agents) + 0.6) if 'C' in a.id else viridis2(i/len(agents)),
                            linewidth=3)

    plt.xlim([df.start.min()-0.5, df.end.max()+0.5])
    plt.ylim([df.offset.min()-0.1, df.offset.max()+0.1])
    # plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper center')
    plt.grid(b=True, axis='x')
    fig.axes[0].get_yaxis().set_visible(False)
    # plt.yticks([a.offset for a in agents], [a.id + " vc=" + str(a.voteCount) for a in agents])
    # plt.xticks(ticks=list(range(5)), labels=["$p$", *["$c_{}$".format(i) for i in range(1,7)]], size=13)
    # plt.xticks(ticks=list(range(5)), labels=["$c_{1}$\n$c_{0}$", "$c_{2}$", "$c_{4}$\n$c_{3}$", "$c_{5}$", "$c_{7}$\n$c_{6}$"], size=13)
    # plt.xticks(ticks=list(range(6)), labels=["$v_{1}$", "$v_{3}$\n$v_{2}$", "$v_{5}$\n$v_{4}$", "$v_{6}$", "$v_{7}$", "$v_{0}$"], size=13)
    # fig.axes[0].get_xticklabels()[4].set_color("dodgerblue")
    # fig.axes[0].get_xticklabels()[5].set_color("dodgerblue")
    # fig.axes[0].get_xticklabels()[6].set_color("dodgerblue")
    plt.tight_layout()
    partialFormatter = partial(formatter, agents)
    # savePath = "/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter2/Figs/alt-ncop-election-example.png"
    # plt.savefig(savePath)
    plt.show()

#%%
def crPlot():
    A = np.array([1,1,1,1,0,
                  1,0,1,0,0,
                  1,1,0,1,0,
                  0,0,0,0,1]).reshape(4, 5)
    C = [Candidate("c0", 2, 2),
         Candidate("c1", 1, 0.5),
         Candidate("c2", 4, 1),
         Candidate("c3", 2, 1.25),
         Candidate("cP", 6, 0.5)]
    V = [Voter("v0", 2, 1),
         Voter("v1", 4, 0.5),
         Voter("v2", 1.5, 0.2),
         Voter("v3", 6, 0.2)]
    return Profile(A, C, V)




#%%
colors = {'cP':'gold',
            'c0':'red', 'c1':'red', 'c2':'red', 'c3':'red',
            'v0':'blue', 'v1':'blue', 'v2':'blue',
            'v3':'lightblue'}

plotVCRAgents(vcrProfileToAgentsWithColors(crPlot(), colors))


#%%
def drawAgentLine(xStart, xEnd, y, label, color):
    p1 = plt.hlines(y=y,
                    xmin=xStart,
                    xmax=xEnd,
                    label=label,
                    color=color,
                    linewidth=3)

    plt.text(x=xStart + (xEnd - xStart)//2, y=y+0.03, s=label, fontdict={'size':9})

    yOff = 0.025
    p2 = plt.vlines(x=xStart,
                    ymin=y - yOff,
                    ymax=y + yOff, 
                    color=color,
                    linewidth=3)

    p3 = plt.vlines(x=xEnd,
                    ymin=y - yOff,
                    ymax=y + yOff, 
                    color=color,
                    linewidth=3)

#%%
def stage1():
    plt.figure(figsize=(4,3))
    plt.title("Stage 1 - P wins")

    drawAgentLine(6, 8, 1, "P", "black")
    drawAgentLine(6.5, 7.5, 0.85, "v1", "black")


    drawAgentLine(1, 4, 1, "c1", "black")
    plt.grid(b=True, axis='both')
    plt.xlim([0,10])
    plt.ylim([0.5,1.5])
    plt.savefig("/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter8/Figs/sc-cc-stage-{}.png".format(1))
    plt.show()

def stage2():
    plt.figure(figsize=(4,3))
    plt.title("Stage 2 - Tie, delete v2 to win")

    drawAgentLine(6, 8, 1, "P", "black")
    drawAgentLine(6.5, 7.5, 0.85, "v1", "black")


    drawAgentLine(1, 4, 1, "c1", "black")
    drawAgentLine(3.5, 4, 0.85, "v2", "black")

    plt.grid(b=True, axis='both')
    plt.xlim([0,10])
    plt.ylim([0.5,1.5])
    plt.savefig("/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter8/Figs/sc-cc-stage-{}.png".format(2))
    plt.show()

def stage3():
    plt.figure(figsize=(4,3))
    plt.title("Stage 3 - c2 not dangerous")

    drawAgentLine(6, 8, 1, "P", "black")
    drawAgentLine(6.5, 7.5, 0.85, "v1", "black")


    drawAgentLine(1, 4, 1, "c1", "black")
    drawAgentLine(3.5, 4, 0.85, "v2", "black")

    drawAgentLine(1.3, 2, 1.15, "c2", "black")


    plt.grid(b=True, axis='both')
    plt.xlim([0,10])
    plt.ylim([0.5,1.5])
    plt.savefig("/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter8/Figs/sc-cc-stage-{}.png".format(3))
    plt.show()

def stage4():
    plt.figure(figsize=(4,3))
    plt.title("Stage 4 - [c1,c2] dangerous, delete [v2,v3]")

    drawAgentLine(6, 8, 1, "P", "black")
    drawAgentLine(6.5, 7.5, 0.85, "v1", "black")


    drawAgentLine(1, 4, 1, "c1", "black")
    drawAgentLine(3.5, 4, 0.85, "v2", "black")

    drawAgentLine(1.3, 2, 1.15, "c2", "black")
    drawAgentLine(1.35, 1.75, 0.85, "v3", "black")

    plt.grid(b=True, axis='both')
    plt.xlim([0,10])
    plt.ylim([0.5,1.5])
    plt.savefig("/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter8/Figs/sc-cc-stage-{}.png".format(4))
    plt.show()

#%%
def stage5():
    plt.figure(figsize=(4,3))
    plt.title("Stage 5 - delete [v2, v4]")

    drawAgentLine(6, 8, 1, "P", "black")
    drawAgentLine(6.5, 7.5, 0.85, "v1", "black")
    drawAgentLine(6.5, 7.5, 0.75, "v6", "black")
    drawAgentLine(6.5, 7.5, 0.65, "v7", "black")



    drawAgentLine(1, 4, 1, "c1", "black")
    drawAgentLine(3.5, 4, 0.85, "v2", "black")

    drawAgentLine(1.3, 2, 1.15, "c2", "black")
    drawAgentLine(1.35, 1.75, 0.85, "v3", "black")

    drawAgentLine(1.5, 3.5, 0.75, "v4", "black")
    drawAgentLine(1.5, 3.5, 0.65, "v5", "black")



    plt.grid(b=True, axis='both')
    plt.xlim([0,10])
    plt.ylim([0.5,1.5])
    plt.savefig("/home/jasiek/Projects/AGH/MGR/master-thesis/Chapter8/Figs/sc-cc-stage-{}.png".format(5))
    plt.show()


stage5()
