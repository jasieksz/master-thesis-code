from numpy import ndarray, array, concatenate
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
    C: List[Candidate]
    V: List[Voter]

    def __str__(self):
        res = str(self.A)
        res += "\n"
        for c in self.C:
            res += c.shortPrint() + "\n"
        for v in self.V:
            res += v.shortPrint() + "\n"
        return res

    def indexProduct(self):
        return list(product(range(len(self.C)), range(len(self.V))))

    def asNumpy(self): # [ C, V, c1x, c1r, ..., cnx, cnr, v1x, v1r, ..., vmx, vmr, A00, ..., Anm ]
        return concatenate([
            array(self.A.shape),
            array([(c.x, c.r) for c in self.C]).flatten(),
            array([(v.x, v.r) for v in self.V]).flatten(),
            self.A.flatten()])    

    @staticmethod
    def fromNumpy(npArray: ndarray):
        C = int(npArray[0])
        V = int(npArray[1])
        candidates = [Candidate(id=chr(ord('A') + i), x=e[0], r=e[1]) for i, e in
                      enumerate(npArray[2:2 + 2 * C].reshape(C, 2))]
        voters = [Voter(id='V' + str(i), x=e[0], r=e[1]) for i, e in
                  enumerate(npArray[2 + 2 * C:2 + 2 * C + 2 * V].reshape(V, 2))]
        approvals = npArray[2 + 2 * C + 2 * V:].reshape(C, V)

        return Profile(A=approvals, C=candidates,  V=voters)

    @staticmethod
    def fromILPRes(approvalSet: ndarray, result, candidatesIds: List[str], votersIds: List[str]):
        voters = [Voter(id=vId, x=result['x'+vId], r=result['r'+vId]) for vId in votersIds]
        candidates = [Candidate(id=cId, x=result['x'+cId], r=result['r'+cId]) for cId in candidatesIds]
        return Profile(A=approvalSet, C=candidates, V=voters)


def notVCR33():
    A = array([1, 0, 1, 1, 1, 0, 0, 1, 1]).reshape(3, 3)
    C = [Candidate("A", 3, 2),
         Candidate("B", 7, 2),
         Candidate("C", 11, 1)]
    V = [Voter("0", 1, 1),
         Voter("1", 5, 1),
         Voter("2", 10, 2)]
    return Profile(A, C, V)


def VCR22():
    A = array([1, 1, 0, 1]).reshape(2, 2)
    C = [Candidate("A", 1, 1),
         Candidate("B", 3, 1)]
    V = [Voter("0", 2, 1),
         Voter("1", 4, 1)]
    return Profile(A, C, V)


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
    return Profile(A, C, V)
