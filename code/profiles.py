#%%
import numpy as np
import pandas as pd
from sympy.utilities.iterables import multiset_permutations
import itertools
import math

#%%
def getAllSquareProfiles(candidate, voter):
    singleVotes = list(map(list, itertools.product([0,1], repeat=candidate)))
    return [np.array(profile) for profile in list(map(list, itertools.product(singleVotes, repeat=voter)))]

def consecutiveOnes1D(A):
    ones = False
    afterOnes = False
    for i in range(len(A)):
        if A[i] == 1 and afterOnes == False:
            ones = True
        elif A[i] == 0 and ones == True:
            afterOnes = True
        elif A[i] == 1 and afterOnes == True:
            return False
    return True

def consecutiveOnes2D(A):
    cr = [consecutiveOnes1D(A[:,i]) for i in range(A.shape[1])] # kolumny
    vr = [consecutiveOnes1D(A[i,:]) for i in range(A.shape[0])] # wiersze
    return all(cr) or all(vr)

def emptyVote2D(A): # contains at least one empty vote (zeros)
    return not all([sum(A[i]) != 0 for i in range(len(A))])

def interestingProfile(profile):
    return not consecutiveOnes2D(profile) and not emptyVote2D(profile)

def shuffleProfile(profile, colSwap, rowSwap):
    SP = np.copy(profile)
    SP[:, list(range(len(colSwap)))] = SP[:,list(colSwap)] # swap candidates (Columns)
    SP[list(range(len(rowSwap)))] = SP[list(rowSwap)] # swap voters (rows)
    return SP

def getProfilePermutations(P):
    dim = P.shape
    C = list(itertools.permutations(list(range(dim[1])), dim[1])) # Candidates / Columns
    V = list(itertools.permutations(list(range(dim[0])), dim[0])) # Voters / Rows
    CxV = list(itertools.product(C, V))
    return [shuffleProfile(profile=P, colSwap=shf[0], rowSwap=shf[1]) for shf in CxV]

def profileDataFrame(A):
    candidates = [chr(ord('A')+i) for i in range(A.shape[1])]
    voters = ['v'+str(i) for i in range(A.shape[0])]
    return pd.DataFrame(A, voters, candidates)

def generatePossibleVCR(candidate, voter):
    P = getAllSquareProfiles(candidate, voter)
    interestingProfiles = [p for p in P if interestingProfile(p)]
    possibleVCRProfiles = [P for P in interestingProfiles if not any([consecutiveOnes2D(p) for p in getProfilePermutations(P)])]
    return possibleVCRProfiles

#%%
P44 = generatePossibleVCR(candidate=4,voter=4)
len(P44)

#%%
P34 = generatePossibleVCR(candidate=3,voter=4)

#%%
P33 = generatePossibleVCR(candidate=3,voter=3)

#%%
for p in P23:
    print("1.\n```")
    print(profileDataFrame(p))
    print("```")

#%%
np.array_equal(P33[0], np.array([0,1,1,1,0,1,1,1,0]).reshape(3,3))

#%%
VCR44 = np.array([1,1,0,0,0,1,1,0,0,1,0,1,1,1,1,1]).reshape(4,4)

#%%
import time

#%%
t = time.process_time()
P44 = generatePossibleVCR(candidate=4,voter=4)
print(elapsed_time)
len(P44)

#%%
any([consecutiveOnes2D(p) for p in VCR44PP])

#%%
with open('P44.npy', 'wb') as f:
    np.save(f, P44)

#%%
A = np.load('P44.npy')

#%%
np.array_equal(A, P44)

#%%
A = np.array([[0, 0, 0, 1],
       [0, 0, 1, 1],
       [1, 0, 1, 0],
       [0, 0, 1, 0]])

any([consecutiveOnes2D(p) for p in getProfilePermutations(A)])