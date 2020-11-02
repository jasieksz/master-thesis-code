#%%
import numpy as np
import pandas as pd
from sympy.utilities.iterables import multiset_permutations
import itertools
import math
from typing import NamedTuple
from typing import List, Union, Optional
from pprint import pprint

#%%
class Candidate(NamedTuple):
    id: str
    x: int
    r: int

    def shortPrint(self):
        return "C_" + self.id + " : [x=" + str(self.x) + ", r=" + str(self.r) + "]"

class Voter(NamedTuple):
    id: str
    x: int
    r: int
    A: List[Candidate]

    def shortPrint(self):
        return "V_" + self.id + " : [x=" + str(self.x) + ", r=" + str(self.r) + "]"

class Profile(NamedTuple):
    V: List[Voter]
    C: List[Candidate]
    def __str__(self):
        res = ""
        for v in self.V:
            res += v.shortPrint() + "\n"
        for c in self.C:
            res += c.shortPrint() + "\n"
        res += "\n"
        res += str(self.approvalSet())
        return res

    def approvalSet(self):
        return np.array(\
            [1 if self.C[c] in self.V[v].A else 0\
                for v in range(len(self.V))\
                for c in range(len(self.C))]\
            ).reshape(len(self.V), len(self.C))

#%%
def getById(collection: List[Union[Voter,Candidate]], id) -> Optional[Union[Voter, Candidate]]:
    for element in collection:
        if (element.id == id):
            return element
    return None
    
def indexProduct(P: Profile):
    return list(itertools.product(range(len(P.V)), range(len(P.C))))

def nextChar(char):
    return chr(ord(char)+1)

def randomInt(start, end):
    return np.random.randint(start, end)

def flatten(collection):
    return list(itertools.chain(*collection))

#%%
def vcrDef(v: Voter, c: Candidate) -> bool:
    return round(math.sqrt((v.x - c.x)**2), 4) <= v.r + c.r 

def isVCR(P: Profile) -> bool:
    perm = indexProduct(P)
    res1 = [(P.V[v], P.C[c]) for (v,c) in perm if vcrDef(P.V[v], P.C[c])]
    res2 = [(P.V[v], P.C[c]) for (v,c) in perm if not vcrDef(P.V[v], P.C[c])]
    return all([c in v.A for (v,c) in res1]) and all([not c in v.A for (v,c) in res2])

#%%
def randomApprovalSet(C: List[Candidate]) -> List[Candidate]:
    return [C[i] for i in set(np.random.choice(range(len(C)), randomInt(1, len(C)+1)))]

def randomProfile(sizeV, sizeC):
    C = [Candidate(chr(ord('A') + i), randomInt(0, 11), randomInt(1,4)) for i in range(sizeC)]
    V = [Voter(str(i), randomInt(0, 11), randomInt(1,4), randomApprovalSet(C)) for i in range(sizeV)]
    return Profile(V,C)

def notVCR33():
    C = [Candidate("A", 3, 2),
        Candidate("B", 7, 2),
        Candidate("C", 11, 1)]

    V = [Voter("0", 1, 1, [C[0], C[2]]),
        Voter("1", 5, 1, [C[0], C[1]]),
        Voter("2", 10, 2, [C[1], C[2]])]

    return Profile(V,C)

def VCR22():
    C = [Candidate("A", 1, 1),
        Candidate("B", 3, 1)]

    V = [Voter("0", 2, 1, [C[0], C[1]]),
        Voter("1", 4, 1, [C[1]])]

    return Profile(V,C)

def VCR44():
    C = [Candidate("A", x=0, r=0.1),
        Candidate("B", x=0.4, r=0.6),
        Candidate("C", x=0.8, r=0.1),
        Candidate("D", x=1.2, r=0.2)]

    V = [Voter("0", x=0.2, r=0.1, A=[C[0], C[1]]),
        Voter("1", x=0.6, r=0.1, A=[C[1], C[2]]),
        Voter("2", x=1, r=0.05, A=[C[1], C[3]]),
        Voter("3", x=0.1, r=1.1, A=[C[0], C[1], C[2], C[3]])]

    return Profile(V,C)

#%%
i = 0
VCRs = []
while(i < 100000):
    P = randomProfile(3,3)
    if (isVCR(P)):
        # if (True or not any([consecutiveOnes2D(p) for p in getProfilePermutations(P.approvalSet())])):
        VCRs.append(P)
    i += 1
