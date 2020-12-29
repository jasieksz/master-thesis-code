from math import sqrt
from code.profiles.definitions import Profile, Voter, Candidate
from itertools import product


def vcrProperty(voter: Voter, candidate: Candidate) -> bool:
    return round(sqrt((voter.x - candidate.x) ** 2), 4) <= voter.r + candidate.r

def isVCR(profile: Profile) -> bool:
    perm = list(product(range(len(profile.C)), range(len(profile.V))))
    for (vIndex, cIndex) in perm:
        if vcrProperty(profile.V[vIndex], profile.C[cIndex]):
            if profile.A[vIndex, cIndex] == 0:
                return False
        else:
            if profile.A[vIndex, cIndex] == 1:
                return False
    return True
