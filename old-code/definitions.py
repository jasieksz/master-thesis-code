import numpy as np
import math
from typing import List, NamedTuple
import itertools

class Candidate(NamedTuple):
    id: str
    x: int
    r: int

    def shortPrint(self):
        return self.id + " : [x=" + str(self.x) + ", r=" + str(self.r) + "]"

class Voter(NamedTuple):
    id: str
    x: int
    r: int

    def shortPrint(self):
        return self.id + " : [x=" + str(self.x) + ", r=" + str(self.r) + "]"

class Profile(NamedTuple):
    A: np.ndarray
    V: List[Voter]
    C: List[Candidate]

    def __str__(self):
        res = str(self.A)
        res += "\n"
        for v in self.V:
            res += v.shortPrint() + "\n"
        for c in self.C:
            res += c.shortPrint() + "\n"
        return res

    def indexProduct(self):
        return list(itertools.product(range(len(self.V)), range(len(self.C))))

def vcrProperty(v: Voter, c: Candidate) -> bool:
    return round(math.sqrt((v.x - c.x)**2), 4) <= v.r + c.r 

def isVCR(P: Profile) -> bool:
    perm = P.indexProduct()
    for (vIndex,cIndex) in perm:
        if (vcrProperty(P.V[vIndex], P.C[cIndex])):
            if (P.A[vIndex, cIndex] == 0):
                return False
        else:
            if (P.A[vIndex, cIndex] == 1):
                return False
    return True