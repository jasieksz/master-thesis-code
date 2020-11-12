#%%
import numpy as np
from profiles import VCR44, notVCR33, VCR22
from definitions import Voter, Candidate, Profile, isVCR
from helpers import consecutiveOnes2D
from detectionILP import detectorMockDist, detectorPosNeg
from pprint import pprint
import math

#%%
def vcrDetectionPosNeg(approvals, voterIds, candidateIds):
    problem, result = detectorPosNeg(approvals, voterIds, candidateIds)

    # Create profile
    voters = [Voter(id=vId, x=result['x'+vId], r=result['r'+vId]) for vId in voterIds]
    candidates = [Candidate(id=cId, x=result['x'+cId], r=result['r'+cId]) for cId in candidateIds]
    profile = Profile(approvals, voters, candidates)

    print("Status : " + str(problem.sol_status) + "\n")
    if (problem.sol_status == -1):
        print([con.name for con in list(problem.constraints.values()) if not con.valid()])
    print(profile)
    print("VCR : " + str(isVCR(profile)))
    return profile, problem


#%% VCR 22
approvals = np.array([1,1,0,1]).reshape(2,2)
voterIds = ['v1', 'v2']
candidateIds = ['A', 'B']
P22, prob = vcrDetectionPosNeg(approvals, voterIds, candidateIds)

#%% VCR 44
approvals = np.array([1,1,1,1,1,1,0,0,0,1,0,1,0,1,1,0]).reshape(4,4)
voterIds = ['v1', 'v2', 'v3', 'v4']
candidateIds = ['A', 'B', 'C', 'D']
P44, prob = vcrDetectionPosNeg(approvals, voterIds, candidateIds)

#%% VCR 33
approvals = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)
voterIds = ['v1', 'v2', 'v3']
candidateIds = ['A', 'B', 'C']
P33, prob = vcrDetectionPosNeg(approvals, voterIds, candidateIds)

#%% NOT 33
approvals = np.array([1,0,1,1,1,0,0,1,1]).reshape(3,3)
voterIds = ['v1', 'v2', 'v3']
candidateIds = ['A', 'B', 'C']
notP33, prob = vcrDetectionPosNeg(approvals, voterIds, candidateIds)

#%% NOT 44
approvals = np.array([1,1,0,1,1,0,1,1,0,1,1,0,0,1,0,1]).reshape(4,4)
voterIds = ['v1', 'v2', 'v3', 'v4']
candidateIds = ['A', 'B', 'C', 'D']
notP44, prob = vcrDetectionPosNeg(approvals, voterIds, candidateIds)
