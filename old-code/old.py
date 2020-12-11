#%%
from pulp import *
import itertools
from typing import List
from definitions import isVCR, Candidate, Voter, Profile
import numpy as np

#%%
##################### detection ILP

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

def mockDistanceVariable():
    prob = LpProblem("vcr", LpMinimize)

    DA1 = LpVariable("DA1",None,None,LpContinuous)
    DA2 = LpVariable("DA2",None,None,LpContinuous)
    DA3 = LpVariable("DA3",None,None,LpContinuous)
    DB1 = LpVariable("DB1",None,None,LpContinuous)
    DB2 = LpVariable("DB2",None,None,LpContinuous)
    DB3 = LpVariable("DB3",None,None,LpContinuous)
    DC1 = LpVariable("DC1",None,None,LpContinuous)
    DC2 = LpVariable("DC2",None,None,LpContinuous)
    DC3 = LpVariable("DC3",None,None,LpContinuous)

    xA = LpVariable("xA",None,None,LpContinuous)
    xB = LpVariable("xB",None,None,LpContinuous)
    xC = LpVariable("xC",None,None,LpContinuous)
    x1 = LpVariable("x1",None,None,LpContinuous)
    x2 = LpVariable("x2",None,None,LpContinuous)
    x3 = LpVariable("x3",None,None,LpContinuous)

    rA = LpVariable("rA",0,None,LpContinuous)
    rB = LpVariable("rB",0,None,LpContinuous)
    rC = LpVariable("rC",0,None,LpContinuous)
    r1 = LpVariable("r1",0,None,LpContinuous)
    r2 = LpVariable("r2",0,None,LpContinuous)
    r3 = LpVariable("r3",0,None,LpContinuous)

#    A B C
#1 | 1 0 0
#2 | 0 1 0
#2 | 0 0 1

    # prob += rA + rB + r1 + r2, "fun"
    prob += 0, "fun"
####
    prob += DA1 == xA - x1, "DA1 pos"
    prob += DA1 == x1 - xA, "DA1 neg"
    prob += DA1 <= rA + r1, "A1 VCR"

    prob += (DA2 == xA - x2) or (DA2 == x2 - xA) , "DA2"
    prob += DA2 >= rA + r2 + 1, "A2 not VCR"

    prob += (DA3 == xA - x3) or (DA3 == x3 - xA) , "DA3"
    prob += DA3 >= rA + r3 + 1, "A3 not VCR"
 ####
    prob += (DB1 == xB - x1) or (DB1 == x1 - xB) , "DB1"
    prob += DB1 >= rB + r1 + 1, "B1 not VCR"

    prob += DB2 == xB - x2, "DB2 pos"
    prob += DB2 == x2 - xB, "DB2 neg"
    prob += DB2 <= rB + r2, "B2 VCR"

    prob += (DB3 == xB - x3) or (DB3 == x3 - xB) , "DB3"
    prob += DB3 >= rB + r3 + 1, "B3 not VCR"
 ####
    prob += (DC1 == xC - x1) or (DC1 == x1 - xC) , "DC1"
    prob += DC1 >= rC + r1 + 1, "C1 not VCR"

    prob += (DC2 == xC - x2) or (DC2 == x2 - xC) , "DC2"
    prob += DC2 >= rC + r2 + 1, "C2 not VCR"

    prob += DC3 == xC - x3, "DC3 pos"
    prob += DC3 == x3 - xC, "DC3 neg"
    prob += DC3 <= rC + r3, "C3 VCR"
 ####

#  x + M * B >= MIN
# -x + M * (1 - B) >= MIN

    prob.solve()
    ilpResult = {var.name : var.varValue for var in prob.variables()}
    return prob, ilpResult

#%%
def maybeVCR33(res):
    A = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)
    C = [
            Candidate("A", res['xA'], res['rA']),
            Candidate("B", res['xB'], res['rB']),
            Candidate("C", res['xC'], res['rC'])
        ]

    V = [
            Voter("1", res['x1'], res['r1']),
            Voter("2", res['x2'], res['r2']),
            Voter("3", res['x3'], res['r3'])
        ]
    return Profile(A,V,C)

#%%
prob, res = mockDistanceVariable()

#%%
print([con.name for con in list(prob.constraints.values()) if not con.valid()])

#%%
prob, res = positiveNegativeConstraint()
print(prob.sol_status)

pprint(res)
print([con.name for con in list(prob.constraints.values()) if not con.valid()])

P = maybeVCR22(res)
print(P)
isVCR(P)


#%%
#%%
def maybeVCR22(res):
    A = np.array([1,1,0,1]).reshape(2,2)
    C = [
            Candidate("A", res['xA'], res['rA']),
            Candidate("B", res['xB'], res['rB']),
        ]

    V = [
            Voter("1", res['x1'], res['r1']),
            Voter("2", res['x2'], res['r2']),
        ]
    return Profile(A,V,C)

def positiveNegativeConstraint22():
    prob = LpProblem("vcr", LpMinimize)

    xA = LpVariable("xA",None,None,LpContinuous)
    xB = LpVariable("xB",None,None,LpContinuous)
    x1 = LpVariable("x1",None,None,LpContinuous)
    x2 = LpVariable("x2",None,None,LpContinuous)

    rA = LpVariable("rA",0,None,LpContinuous)
    rB = LpVariable("rB",0,None,LpContinuous)
    r1 = LpVariable("r1",0,None,LpContinuous)
    r2 = LpVariable("r2",0,None,LpContinuous)


    prob += r1 + r2, "fun"

    prob += xA - x1 <= rA + r1, "A1 VCR pos"
    prob += xA - x1 >= -(rA + r1), "A1 VCR neg"

    prob += (xA - x2 >= rA + r2 + 1) or (xA - x2 <= -(rA + r2 + 1)) , "A2 not VCR pos"


    prob += xB - x1 <= rB + r1, "B1 VCR pos"
    prob += xB - x1 >= -(rB + r1), "B1 VCR neg"

    prob += xB - x2 <= rB + r2, "B2 VCR pos"
    prob += xB - x2 >= -(rB + r2), "B2 VCR neg"


    prob.solve()
    ilpResult = {var.name : var.varValue for var in prob.variables()}
    print(prob.sol_status)
    return prob, ilpResult

#%%
def positiveNegativeConstraint33():
    prob = LpProblem("vcr", LpMinimize)

    xA = LpVariable("xA",-1,10,LpContinuous)
    xB = LpVariable("xB",-1,10,LpContinuous)
    xC = LpVariable("xC",-1,10,LpContinuous)
    x1 = LpVariable("x1",-1,10,LpContinuous)
    x2 = LpVariable("x2",-1,10,LpContinuous)
    x3 = LpVariable("x3",-1,10,LpContinuous)

    rA = LpVariable("rA",1,10,LpContinuous)
    rB = LpVariable("rB",1,10,LpContinuous)
    rC = LpVariable("rC",1,10,LpContinuous)
    r1 = LpVariable("r1",1,10,LpContinuous)
    r2 = LpVariable("r2",1,10,LpContinuous)
    r3 = LpVariable("r3",1,10,LpContinuous)


    prob += r1 + r2 + r3, "fun"

    prob += xA - x1 <= rA + r1, "A1 VCR pos"
    prob += xA - x1 >= -(rA + r1), "A1 VCR neg"
    prob += ((xA - x2 >= rA + r2 + 1) or (xA - x2 <= -(rA + r2 + 1))) , "A2 not VCR"
    prob += ((xA - x3 >= rA + r3 + 1) or (xA - x3 <= -(rA + r3 + 1))) , "A3 not VCR"


    prob += ((xB - x1 >= rB + r1 + 1) or (xB - x1 <= -(rB + r1 + 1))) , "B1 not VCR"
    prob += xB - x2 <= rB + r2, "B2 VCR pos"
    prob += xB - x2 >= -(rB + r2), "B2 VCR neg"
    prob += ((xB - x3 >= rB + r3 + 1) or (xB - x3 <= -(rB + r3 + 1))) , "B3 not VCR"

    prob += ((xC - x1 >= rC + r1 + 1) or (xC - x1 <= -(rC + r1 + 1))) , "C1 not VCR"
    prob += ((xC - x2 >= rC + r2 + 1) or (xC - x2 <= -(rC + r2 + 1))) , "C2 not VCR"
    prob += xC - x3 <= rC + r3, "C3 VCR pos"
    prob += xC - x3 >= -(rC + r3), "C3 VCR neg"

    prob.solve()
    ilpResult = {var.name : var.varValue for var in prob.variables()}
    print(prob.sol_status)
    return prob, ilpResult

#%%
def positiveNegativeConstraint33MIT():
    prob = LpProblem("vcr", LpMinimize)

    xA = LpVariable("xA",-1,10,LpContinuous)
    xB = LpVariable("xB",-1,10,LpContinuous)
    xC = LpVariable("xC",-1,10,LpContinuous)
    x1 = LpVariable("x1",-1,10,LpContinuous)
    x2 = LpVariable("x2",-1,10,LpContinuous)
    x3 = LpVariable("x3",-1,10,LpContinuous)

    rA = LpVariable("rA",1,10,LpContinuous)
    rB = LpVariable("rB",1,10,LpContinuous)
    rC = LpVariable("rC",1,10,LpContinuous)
    r1 = LpVariable("r1",1,10,LpContinuous)
    r2 = LpVariable("r2",1,10,LpContinuous)
    r3 = LpVariable("r3",1,10,LpContinuous)


    prob += r1 + r2 + r3, "fun"

######
    prob += xA - x1 <= rA + r1, "A1 VCR pos"
    prob += -(xA - x1) <= rA + r1, "A1 VCR neg"

    prob += xA - x2 >= rA + r2 + 1, "A2 not VCR pos"
    prob += -(xA - x2) >= rA + r2 + 1 , "A2 not VCR neg"

    prob += xA - x3 >= rA + r3 + 1, "A3 not VCR pos"
    prob += -(xA - x3) >= rA + r3 + 1 , "A3 not VCR neg"

######
    prob += xB - x1 >= rB + r1 + 1, "B1 not VCR pos"
    prob += -(xB - x1) >= rB + r1 + 1, "B1 not VCR neg"

    prob += xB - x2 <= rB + r2, "B2 VCR pos"
    prob += -(xB - x2) <= rB + r2 , "B2 VCR neg"

    prob += xB - x3 >= rB + r3 + 1, "B3 not VCR pos"
    prob += -(xB - x3) >= rB + r3 + 1 , "B3 not VCR neg"

######
    prob += xC - x1 >= rC + r1 + 1, "C1 not VCR pos"
    prob += -(xC - x1) >= rC + r1 + 1, "C1 not VCR neg"

    prob += xC - x2 >= rC + r2 + 1, "C2 Vnot CR pos"
    prob += -(xC - x2) >= rC + r2 + 1 , "C2 not VCR neg"

    prob += xC - x3 <= rC + r3, "C3 VCR pos"
    prob += -(xC - x3) <= rC + r3, "C3 VCR neg"


    prob.solve()
    ilpResult = {var.name : var.varValue for var in prob.variables()}
    print(prob.sol_status)
    return prob, ilpResult

#%%
prob, res = positiveNegativeConstraint33MIT()

#%%
[con.name for con in list(prob.constraints.values()) if not con.valid()]

#%%
isVCR(maybeVCR22(res))

#%%




######################## DETECTION

#%%
import numpy as np
import pandas as pd
from sympy.utilities.iterables import multiset_permutations
import itertools
import math
from typing import NamedTuple
from typing import List, Union, Optional
from pprint import pprint
import time 

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
    
#%%
def vcrNumDef(v: Voter, c: Candidate) -> float:
    return round(math.sqrt((v.x - c.x)**2), 4) - (v.r + c.r) 

def isVCR(P: Profile) -> bool:
    perm = indexProduct(P)
    vcrPairs = []
    notVCRPairs = []
    for (v,c) in perm:
        if vcrDef(P.V[v], P.C[c]):
            if (P.C[c] not in P.V[v].A):
                return False
        else:
            if (P.C[c] in P.V[v].A):
                return False
    return True

def isVCRslow(P: Profile) -> bool:
    perm = indexProduct(P)
    vcrPairs = []
    notVCRPairs = []
    for (v,c) in perm:
        if vcrDef(P.V[v], P.C[c]):
            vcrPairs.append((P.V[v], P.C[c]))
        else:
            notVCRPairs.append((P.V[v], P.C[c]))
    return all([c in v.A for (v,c) in vcrPairs]) and all([not c in v.A for (v,c) in notVCRPairs])

#%%
def randomApprovalSet(C: List[Candidate]) -> List[Candidate]:
    return [C[i] for i in set(np.random.choice(range(len(C)), randomInt(1, len(C)+1)))]

def randomProfile(sizeV, sizeC):
    C = [Candidate(chr(ord('A') + i), randomInt(0, 11), randomInt(1,4)) for i in range(sizeC)]
    V = [Voter(str(i), randomInt(0, 11), randomInt(1,4), randomApprovalSet(C)) for i in range(sizeV)]
    return Profile(V,C)



####################################################3

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



################################################3
for i in range(10):
    name = 'P55-' + str(i) + '.npy'
    print(name)
    with open(basePath+name, 'wb') as f:
        start = window * i
        end = window * (i+1)
        if (i == 9):
            np.save(f, allP55[start:])
        else:
            np.save(f, allP55[start:end])
