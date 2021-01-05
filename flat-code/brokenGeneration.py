#%%
import multiprocessing
from numpy import array, copy, ndarray, save
from numpy.random import default_rng, SeedSequence
import numpy as np
from itertools import product, permutations
from typing import List
from functools import partial
from time import time

#%%
def parallelGenerateApprovalCombinations(cpu: int, C: int, V: int):
    gen = partial(decToArray, C, V)
    with multiprocessing.Pool(cpu) as pool:
        profilesAsNp = list(pool.map(gen, range(2**(C*V))))
    return np.array(profilesAsNp)

def decToArray(C:int, V:int, val:int) -> np.ndarray:
    return np.array(list(bin(val)[2:].rjust(C*V, '0')), dtype=np.int).reshape(C,V)

def parallelGenerateRandomApprovals(cpu: int, count:int, C: int, V: int, prob0:float, prob1:float):
    seq = SeedSequence(1327)
    random_generators = [default_rng(s) for s in seq.spawn(16)]
    gen = partial(generateRandomApproval, random_generators, C, V, prob0, prob1)

    with multiprocessing.Pool(cpu) as pool:
        profilesAsNp = list(pool.map(gen, range(count)))
    return np.array(profilesAsNp)

def generateRandomApproval(random_generators, C:int, V:int, prob0:int, prob1:int, threadIndex) -> ndarray:
    return random_generators[threadIndex%16].choice([0, 1], C*V, p=[prob0, prob1]).reshape(C, V)

def parallelProfileGeneration(cpu: int, candidatesNumber: int, voterNumber: int):
    with multiprocessing.Pool(cpu) as pool:
        singleVotes = list(pool.map(list, product([0, 1], repeat=candidatesNumber)))
        profiles = list(pool.map(list, product(singleVotes, repeat=voterNumber)))
        profilesAsNp = list(pool.map(array, profiles))
    return array(profilesAsNp)


def shuffleProfile(profile: ndarray, colSwap: List[int], rowSwap: List[int]) -> ndarray:
    SP = copy(profile)
    SP[:, list(range(len(colSwap)))] = SP[:, list(colSwap)]  # swap candidates (Columns)
    SP[list(range(len(rowSwap)))] = SP[list(rowSwap)]  # swap voters (rows)
    return SP


def getProfilePermutations(profile: ndarray) -> List[ndarray]:
    dim = profile.shape
    C = list(permutations(list(range(dim[1])), dim[1]))  # Candidates / Columns
    V = list(permutations(list(range(dim[0])), dim[0]))  # Voters / Rows
    CxV = list(product(C, V))
    return [shuffleProfile(profile=profile, colSwap=shf[0], rowSwap=shf[1]) for shf in CxV]


#%%

C = 4
V = 6
# subSet = 1
Ps = parallelGenerateApprovalCombinations(16,C,V)
path = "resources/input/{}C{}V/P{}{}-{}.npy"
w = Ps.shape[0] // 32
print(Ps.shape)

for i in range(32):
    with open(path.format(C,V,C,V,i), 'wb') as f:
        save(f, Ps[i*w:(i+1)*w])

