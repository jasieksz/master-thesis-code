from numpy import ndarray, array
from typing import List, NamedTuple
from itertools import product


class Candidate(NamedTuple):
    id: str
    x: float
    r: float

    def shortPrint(self):
        return self.id + " : [x=" + str(self.x) + ", r=" + str(self.r) + "]"


class Voter(NamedTuple):
    id: str
    x: float
    r: float

    def shortPrint(self):
        return self.id + " : [x=" + str(self.x) + ", r=" + str(self.r) + "]"


class Profile(NamedTuple):
    A: ndarray
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
        return list(product(range(len(self.V)), range(len(self.C))))


def notVCR33():
    A = array([1, 0, 1, 1, 1, 0, 0, 1, 1]).reshape(3, 3)
    C = [Candidate("A", 3, 2),
         Candidate("B", 7, 2),
         Candidate("C", 11, 1)]
    V = [Voter("0", 1, 1),
         Voter("1", 5, 1),
         Voter("2", 10, 2)]
    return Profile(A, V, C)


def VCR22():
    A = array([1, 1, 0, 1]).reshape(2, 2)
    C = [Candidate("A", 1, 1),
         Candidate("B", 3, 1)]
    V = [Voter("0", 2, 1),
         Voter("1", 4, 1)]
    return Profile(A, V, C)


def VCR44():
    A = array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1]).reshape(4, 4)
    C = [Candidate("A", x=0, r=0.1),
         Candidate("B", x=0.4, r=0.6),
         Candidate("C", x=0.8, r=0.1),
         Candidate("D", x=1.2, r=0.2)]
    V = [Voter("0", x=0.2, r=0.1),
         Voter("1", x=0.6, r=0.1),
         Voter("2", x=1, r=0.05),
         Voter("3", x=0.1, r=1.1)]
    return Profile(A, V, C)
