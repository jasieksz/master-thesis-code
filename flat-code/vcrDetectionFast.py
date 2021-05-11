#%%
import gurobipy as gp
from itertools import product
from typing import List
from numpy import ndarray


# %%
def createGPEnv():
    env = gp.Env(empty=True)
    # env.setParam('OutputFlag', 0)
    env.start()
    return env


# %%
def detectVCRPropertyFast(A: ndarray, V:int, C:int, env=None):
    if env is None:
        env = createGPEnv()

    with gp.Model(env=env) as model:

        xV = model.addVars(V, vtype=gp.GRB.CONTINUOUS, name="xV")
        rV = model.addVars(V, lb=0, vtype=gp.GRB.CONTINUOUS, name="rV")

        xC = model.addVars(C, vtype=gp.GRB.CONTINUOUS, name="xC")
        rC = model.addVars(C, lb=0, vtype=gp.GRB.CONTINUOUS, name="rC")
        
        rSums = model.addVars(V,C, vtype=gp.GRB.CONTINUOUS, name="rVC")
        xDiffs = model.addVars(V,C, vtype=gp.GRB.CONTINUOUS, name="xVC")
        xAbsDiffs = model.addVars(V,C, vtype=gp.GRB.CONTINUOUS, name="xVCAbs")
        Z = model.addVars(V,C,2, vtype=gp.GRB.BINARY, name="zVC")

        for (v,c) in product(range(V), range(C)):
            model.addConstr(xDiffs[v,c] == xC[c] - xV[v])
            model.addConstr(rSums[v,c] == rC[c] + rV[v])

            if A[v, c] == 1:
                model.addConstr(xDiffs[v,c] <= rSums[v,c])
                model.addConstr((-xDiffs[v,c]) <= rSums[v,c])
            else:
                model.addConstr(
                    (Z[v,c,0] == 1) >> (xDiffs[v,c] >= rSums[v,c] + 1)
                )
                model.addConstr(
                    (Z[v,c,1] == 1) >> ((-xDiffs[v,c]) >= rSums[v,c] + 1)
                )
                model.addConstr(Z[v,c,0] + Z[v,c,1] >= 1)

        model.optimize()
        model.computeIIS()
        model.write("model.ilp")

        if model.Status == 2:
            return model.Status, {v.varName: v.X for v in model.getVars() if 'r' in v.varName or 'x' in v.varName}
        else:
            return model.Status, {}

#%%
cIds = ['C' + str(i) for i in range(4)]
vIds = ['V' + str(i) for i in range(4)]
env = createGPEnv()
P = VCR44()


#%%
s,d = detectVCRPropertyFast(P.A, 4, 4, env)






#%%
from definitions import Profile, Voter, Candidate

def fromILPRes(approvalSet: ndarray, result, V:int, C:int):
    voters = [Voter(id=vId, x=result['xV[{}]'.format(vId)], r=result['rV[{}]'.format(vId)]) for vId in range(V)]
    candidates = [Candidate(id=cId, x=result['xC[{}]'.format(cId)], r=result['rC[{}]'.format(cId)]) for cId in range(C)]
    return Profile(A=approvalSet, C=candidates, V=voters)


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

#%%
import numpy as np
def notVCR33():
    A = np.array([1, 0, 1, 1, 1, 0, 0, 1, 1]).reshape(3, 3)
    C = [Candidate("A", 3, 2),
         Candidate("B", 7, 2),
         Candidate("C", 11, 1)]
    V = [Voter("0", 1, 1),
         Voter("1", 5, 1),
         Voter("2", 10, 2)]
    return Profile(A, C, V)


def VCR22():
    A = np.array([1, 1, 0, 1]).reshape(2, 2)
    C = [Candidate("A", 1, 1),
         Candidate("B", 3, 1)]
    V = [Voter("0", 2, 1),
         Voter("1", 4, 1)]
    return Profile(A, C, V)


def VCR44():
    A = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1]).reshape(4, 4)
    C = [Candidate("A", x=0, r=0.1),
         Candidate("B", x=0.4, r=0.6),
         Candidate("C", x=0.8, r=0.1),
         Candidate("D", x=1.2, r=0.2)]
    V = [Voter("0", x=0.2, r=0.1),
         Voter("1", x=0.6, r=0.1),
         Voter("2", x=1, r=0.05),
         Voter("3", x=0.1, r=1.1)]
    return Profile(A, C, V)
