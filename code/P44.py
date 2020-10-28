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
profiles = np.load('profiles/P44.npy')
profiles.shape

#%%
def profileDataFrame(A):
    candidates = [chr(ord('A')+i) for i in range(A.shape[1])]
    voters = ['v'+str(i) for i in range(A.shape[0])]
    return pd.DataFrame(A, voters, candidates)

#%%
index = np.random.randint(0,7176)
# index = 5643
P = profiles[index]

if(sum(sum(P)) == 9 and True):
    print(index)
    print(profileDataFrame(P))

    G = nx.Graph()
    V = [0, 1, 2, 3]
    C = ['A', 'B', 'C', 'D']
    G.add_nodes_from(C)
    G.add_nodes_from(V)

    for voter in range(len(V)):
        for cand in range(len(C)):
            if (P[voter][cand] == 1):
                G.add_edge(V[voter], C[cand])

    # G.remove_node('A')
    pos = nx.spring_layout(G)
    # nx.draw_networkx(G, pos, node_color=['r','r','r','r','g','g','g','g'][1:])
    nx.draw_networkx(G, pos, node_color=['r','r','r','r','g','g','g','g'][:8])


#%%
VCR44 = np.array([1,1,0,0,0,1,1,0,0,1,0,1,1,1,1,1]).reshape(4,4)
print(profileDataFrame(VCR44))

#%%
CP = np.copy(P)
CP = np.delete(CP, 2, axis=0)
print(profileDataFrame(CP))

#%%
VCR44 = np.array([1,1,0,0,0,1,1,0,0,1,0,1,1,1,1,1]).reshape(4,4)
print(profileDataFrame(VCR44))
CVCR44 = np.copy(VCR44)
CVCR44 = np.delete(CVCR44, 1, axis=1) #cols
CVCR44 = np.delete(CVCR44, 3, axis=0) #rows
print(profileDataFrame(CVCR44))
