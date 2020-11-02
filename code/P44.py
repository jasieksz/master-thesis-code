#%%
import numpy as np
import pandas as pd
from sympy.utilities.iterables import multiset_permutations
import itertools
import math
import importlib
import networkx as nx
from matplotlib import pyplot as plt

#%%
def profileDataFrame(A):
    candidates = [chr(ord('A')+i) for i in range(A.shape[1])]
    voters = ['v'+str(i) for i in range(A.shape[0])]
    return pd.DataFrame(A, voters, candidates)

def delCand(profile, candIndex):
    return np.delete(profile, candIndex, axis=1)

def delVote(profile, voteIndex):
    return np.delete(profile, voteIndex, axis=0)

def cleanProfile(profile, delVoters, delCands):
    for voter in delVoters:
        profile = delVote(profile, voter)
    for cand in delCands:
        profile = delCand(profile, cand)
    return profile

def drawGraph(profile):
    dim = profile.shape
    G = nx.Graph()

    V = list(range(dim[0]))
    G.add_nodes_from(V)

    C = [chr(ord('A')+i) for i in range(dim[1])] 
    G.add_nodes_from(C)

    for voter in range(dim[0]):
        for cand in range(dim[1]):
            if (profile[voter][cand] == 1):
                G.add_edge(V[voter], C[cand])

    pos = nx.spring_layout(G)
    nodeColors = ['g' for i in range(dim[0])] + ['r' for i in range(dim[1])] # V0, ... Vn, C0, ..., Cm
    nx.draw_networkx(G, pos, node_color=nodeColors)
    plt.show()

#%%
profiles = np.load('../profiles/P44.npy')
profiles.shape

#%%
index = np.random.randint(0,7176)
P = profiles[index]
if(sum(sum(P)) >= 9):
    print(index)
    print(profileDataFrame(P))
    drawGraph(P)

#%%
cleanedP = cleanProfile(P, delVoters=[3], delCands=[])
print(profileDataFrame(cleanedP))
drawGraph(cleanedP)



#%%
VCR44 = np.array([1,1,0,0,0,1,1,0,0,1,0,1,1,1,1,1]).reshape(4,4)
print(profileDataFrame(VCR44))
drawGraph(VCR44)
CVCR = cleanProfile(VCR44, [3], [1])
print(profileDataFrame(CVCR))
drawGraph(CVCR)

