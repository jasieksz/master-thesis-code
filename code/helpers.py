import numpy as np
import pandas as pd
from sympy.utilities.iterables import multiset_permutations
import itertools
import math

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