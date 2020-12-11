import numpy as np
from definitions import Candidate, Voter, Profile

def notVCR33():
    A = np.array([1,0,1,1,1,0,0,1,1]).reshape(3,3)
    C = [Candidate("A", 3, 2),
        Candidate("B", 7, 2),
        Candidate("C", 11, 1)]
    V = [Voter("0", 1, 1),
        Voter("1", 5, 1),
        Voter("2", 10, 2)]
    return Profile(A,V,C)

def VCR22():
    A = np.array([1,1,0,1]).reshape(2,2)
    C = [Candidate("A", 1, 1),
        Candidate("B", 3, 1)]
    V = [Voter("0", 2, 1),
        Voter("1", 4, 1)]
    return Profile(A,V,C)

def VCR44():
    A = np.array([1,1,0,0,0,1,1,0,0,1,0,1,1,1,1,1]).reshape(4,4)
    C = [Candidate("A", x=0, r=0.1),
        Candidate("B", x=0.4, r=0.6),
        Candidate("C", x=0.8, r=0.1),
        Candidate("D", x=1.2, r=0.2)]
    V = [Voter("0", x=0.2, r=0.1),
        Voter("1", x=0.6, r=0.1),
        Voter("2", x=1, r=0.05),
        Voter("3", x=0.1, r=1.1)]
    return Profile(A,V,C)