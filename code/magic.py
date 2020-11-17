import numpy as np
from profiles import VCR44, notVCR33, VCR22
from definitions import Voter, Candidate, Profile, isVCR
from helpers import consecutiveOnes2D
from detectionILP import detectorMockDist, detectorPosNeg
from pprint import pprint
import math
import time

def vcrDetectionPosNeg(approvals, voterIds, candidateIds):
    problem, result = detectorPosNeg(approvals, voterIds, candidateIds)

    # Create profile
    voters = [Voter(id=vId, x=result['x'+vId], r=result['r'+vId]) for vId in voterIds]
    candidates = [Candidate(id=cId, x=result['x'+cId], r=result['r'+cId]) for cId in candidateIds]
    profile = Profile(approvals, voters, candidates)

    if (problem.sol_status == 1):
        return isVCR(profile), profile
    else:
        return (False, profile)


def vcr44Detection(A):
    return vcrDetectionPosNeg(A, ['v1' ,'v2', 'v3', 'v4'], ['A', 'B', 'C', 'D'])

def vcr55Detection(A):
    return vcrDetectionPosNeg(A, ['v1' ,'v2', 'v3', 'v4', 'v5'], ['A', 'B', 'C', 'D', 'E'])

def vcr66Detection(A):
    return vcrDetectionPosNeg(A, ['v1' ,'v2', 'v3', 'v4', 'v5', 'v6'], ['A', 'B', 'C', 'D', 'E', 'F'])

def npArray(A):
    return np.array(A)