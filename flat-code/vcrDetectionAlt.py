#%%
import gurobipy as gp
from itertools import product
from typing import List
from time import time
from numpy import ndarray
from typing import NamedTuple


# %%
def createGPEnv():
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    return env

#%%
def createBaseModel(C: List[str], V: List[str], env=None):
    indexPairs = list(product(range(len(C)), range(len(V))))

    if env is None:
        env = createGPEnv()

    with gp.Model(env=env) as model:
        # Xa variable - positon of agent
        positions = {
            "x" + agent: model.addVar(vtype=gp.GRB.CONTINUOUS, name="x" + agent) for agent in C + V
        }

        # Ra variable - radius of agent
        radiuses = {
            "r" + agent: model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="r" + agent) for agent in C + V
        }

        Z1Vars = {
            C[c] + V[v]: model.addVar(vtype=gp.GRB.BINARY, name="Z1" + C[c] + V[v]) for c, v in indexPairs
        }

        Z2Vars = {
            C[c] + V[v]: model.addVar(vtype=gp.GRB.BINARY, name="Z2" + C[c] + V[v]) for c, v in indexPairs
        }

        model.write('resources/models/vcr-40C40V-base.mps')
# candidatesIds = ['C' + str(i) for i in range(40)]
# votersIds = ['V' + str(i) for i in range(40)]
# createBaseModel(candidatesIds, votersIds)


#%%
def detectVCRPropertyWithModel(A: ndarray, C: List[str], V: List[str], modelPath, env=None):
    indexPairs = list(product(range(len(C)), range(len(V))))

    if env is None:
        env = createGPEnv()

    with gp.read(modelPath) as model:
        modelVar = {v.varName: v for v in model.getVars()}
        # c,v - agent ID (name), cI, vI - agent index
        for (c, v), (cI, vI) in zip(product(C, V), product(range(len(C)), range(len(V)))):
            if A[vI, cI] == 1:
                model.addConstr(modelVar['x' + c] - modelVar['x' + v] <= modelVar['r' + c] + modelVar['r' + v],
                                "VCR-pos-" + c + v)
                model.addConstr(-(modelVar['x' + c] - modelVar['x' + v]) <= modelVar['r' + c] + modelVar['r' + v],
                                "VCR-neg-" + c + v)
            else:
                model.addConstr(
                    (modelVar['Z1' + c + v] == 1) >> ((modelVar['x' + c] - modelVar['x' + v]) >= modelVar['r' + c] + modelVar['r' + v] + 1)
                )
                model.addConstr(
                    (modelVar['Z2' + c + v] == 1) >> ((-modelVar['x' + c] + modelVar['x' + v]) >= modelVar['r' + c] + modelVar['r' + v] + 1)
                )

                model.addConstr(modelVar['Z1' + c + v] + modelVar['Z2' + c + v] >= 1)

        model.setParam('OutputFlag', False)
        model.optimize()

        if model.Status == 2:
            return model.Status, {v.varName: v.X for v in model.getVars() if 'r' in v.varName or 'x' in v.varName}
        else:
            return model.Status, {}

# %%
def detectVCRProperty(A: ndarray, C: List[str], V: List[str], env=None):
    indexPairs = list(product(range(len(C)), range(len(V))))

    if env is None:
        env = createGPEnv()

    with gp.Model(env=env) as model:
        # Xa variable - positon of agent
        positions = {
            "x" + agent: model.addVar(vtype=gp.GRB.CONTINUOUS, name="x" + agent) for agent in C + V
        }

        # Ra variable - radius of agent
        radiuses = {
            "r" + agent: model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="r" + agent) for agent in C + V
        }

        Z1Vars = {
            C[c] + V[v]: model.addVar(vtype=gp.GRB.BINARY, name="Z1" + C[c] + V[v]) for c, v in indexPairs
        }

        Z2Vars = {
            C[c] + V[v]: model.addVar(vtype=gp.GRB.BINARY, name="Z2" + C[c] + V[v]) for c, v in indexPairs
        }

        # c,v - agent ID (name), cI, vI - agent index
        for (c, v), (cI, vI) in zip(product(C, V), product(range(len(C)), range(len(V)))):
            if A[vI, cI] == 1:
                model.addConstr(positions['x' + c] - positions['x' + v] <= radiuses['r' + c] + radiuses['r' + v],
                                "VCR-pos-" + c + v)
                model.addConstr(-(positions['x' + c] - positions['x' + v]) <= radiuses['r' + c] + radiuses['r' + v],
                                "VCR-neg-" + c + v)
            else:
                model.addConstr(
                    (Z1Vars[c + v] == 1) >> ((positions['x' + c] - positions['x' + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1)
                )
                model.addConstr(
                    (Z2Vars[c + v] == 1) >> ((-positions['x' + c] + positions['x' + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1)
                )

                model.addConstr(Z1Vars[c + v] + Z2Vars[c + v] >= 1)

        model.setParam('OutputFlag', False)
        model.optimize()

        if model.Status == 2:
            return model.Status, {v.varName: v.X for v in model.getVars() if 'r' in v.varName or 'x' in v.varName}
        else:
            return model.Status, {}


# %%
def findVRPoints(A: ndarray, C: List[str], V: List[str], env):  # CR = 0
    indexPairs = list(product(range(len(C)), range(len(V))))

    if env is None:
        env = createGPEnv()

    with gp.Model(env=env) as model:
        # Xa variable - positon of agent
        positions = {
            "x" + agent: model.addVar(vtype=gp.GRB.CONTINUOUS, name="x" + agent) for agent in C + V
        }

        # Ra variable - radius of agent
        radiuses = {
            "r" + agent: model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="r" + agent) for agent in C + V
        }

        Z1Vars = {
            C[c] + V[v]: model.addVar(vtype=gp.GRB.BINARY, name="Z1" + C[c] + V[v]) for c, v in indexPairs
        }

        Z2Vars = {
            C[c] + V[v]: model.addVar(vtype=gp.GRB.BINARY, name="Z2" + C[c] + V[v]) for c, v in indexPairs
        }

        # c,v - agent ID (name), cI, vI - agent index
        for c in C:
            model.addConstr(radiuses['r' + c] == 0, "VR-" + c)
        for (c, v), (cI, vI) in zip(product(C, V), product(range(len(C)), range(len(V)))):
            if A[vI, cI] == 1:
                model.addConstr(positions['x' + c] - positions['x' + v] <= radiuses['r' + c] + radiuses['r' + v],
                                "VCR-pos-" + c + v)
                model.addConstr(-(positions['x' + c] - positions['x' + v]) <= radiuses['r' + c] + radiuses['r' + v],
                                "VCR-neg-" + c + v)
            else:
                model.addConstr(
                    (Z1Vars[c + v] == 1) >> ((positions['x' + c] - positions['x' + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1)
                )
                model.addConstr(
                    (Z2Vars[c + v] == 1) >> ((-positions['x' + c] + positions['x' + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1)
                )

                model.addConstr(Z1Vars[c + v] + Z2Vars[c + v] >= 1)

        model.setParam('OutputFlag', False)
        model.optimize()

        if model.Status == 2:
            return True, {v.varName: v.X for v in model.getVars() if 'r' in v.varName or 'x' in v.varName}
        else:
            return False, {"status": model.status}

# %%
class C1PResult(NamedTuple):
    status:bool
    time:float

def detectVRProperty(A: ndarray, C: List[str], V: List[str], env):  # CR = 0
    indexPairs = list(product(range(len(C)), range(len(V))))

    if env is None:
        env = createGPEnv()

    startTime = time()
    with gp.Model(env=env) as model:
        # Xa variable - positon of agent
        positions = {
            "x" + agent: model.addVar(vtype=gp.GRB.CONTINUOUS, name="x" + agent) for agent in C + V
        }

        # Ra variable - radius of agent
        radiuses = {
            "r" + agent: model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="r" + agent) for agent in C + V
        }

        Z1Vars = {
            C[c] + V[v]: model.addVar(vtype=gp.GRB.BINARY, name="Z1" + C[c] + V[v]) for c, v in indexPairs
        }

        Z2Vars = {
            C[c] + V[v]: model.addVar(vtype=gp.GRB.BINARY, name="Z2" + C[c] + V[v]) for c, v in indexPairs
        }

        # c,v - agent ID (name), cI, vI - agent index
        for c in C:
            model.addConstr(radiuses['r' + c] == 0, "VR-" + c)
        for (c, v), (cI, vI) in zip(product(C, V), product(range(len(C)), range(len(V)))):
            if A[vI, cI] == 1:
                model.addConstr(positions['x' + c] - positions['x' + v] <= radiuses['r' + c] + radiuses['r' + v],
                                "VCR-pos-" + c + v)
                model.addConstr(-(positions['x' + c] - positions['x' + v]) <= radiuses['r' + c] + radiuses['r' + v],
                                "VCR-neg-" + c + v)
            else:
                model.addConstr(
                    (Z1Vars[c + v] == 1) >> ((positions['x' + c] - positions['x' + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1)
                )
                model.addConstr(
                    (Z2Vars[c + v] == 1) >> ((-positions['x' + c] + positions['x' + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1)
                )

                model.addConstr(Z1Vars[c + v] + Z2Vars[c + v] >= 1)

        model.setParam('OutputFlag', False)
        model.optimize()
        totalTime = time() - startTime

        return C1PResult(status = True if model.Status == 2 else False, time=totalTime)

# %%
def detectCRProperty(A: ndarray, C: List[str], V: List[str], env):  # voter radius = 0
    indexPairs = list(product(range(len(C)), range(len(V))))

    if env is None:
        env = createGPEnv()

    with gp.Model(env=env) as model:
        # Xa variable - positon of agent
        positions = {
            "x" + agent: model.addVar(vtype=gp.GRB.CONTINUOUS, name="x" + agent) for agent in C + V
        }

        # Ra variable - radius of agent
        radiuses = {
            "r" + agent: model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="r" + agent) for agent in C + V
        }

        Z1Vars = {
            C[c] + V[v]: model.addVar(vtype=gp.GRB.BINARY, name="Z1" + C[c] + V[v]) for c, v in indexPairs
        }

        Z2Vars = {
            C[c] + V[v]: model.addVar(vtype=gp.GRB.BINARY, name="Z2" + C[c] + V[v]) for c, v in indexPairs
        }

        # c,v - agent ID (name), cI, vI - agent index
        for v in V:
            model.addConstr(radiuses['r' + v] == 0, "CR-" + v)
        for (c, v), (cI, vI) in zip(product(C, V), product(range(len(C)), range(len(V)))):
            if A[vI, cI] == 1:
                model.addConstr(positions['x' + c] - positions['x' + v] <= radiuses['r' + c] + radiuses['r' + v],
                                "VCR-pos-" + c + v)
                model.addConstr(-(positions['x' + c] - positions['x' + v]) <= radiuses['r' + c] + radiuses['r' + v],
                                "VCR-neg-" + c + v)
            else:
                model.addConstr(
                    (Z1Vars[c + v] == 1) >> ((positions['x' + c] - positions['x' + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1)
                )
                model.addConstr(
                    (Z2Vars[c + v] == 1) >> ((-positions['x' + c] + positions['x' + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1)
                )

                model.addConstr(Z1Vars[c + v] + Z2Vars[c + v] >= 1)

        model.setParam('OutputFlag', False)
        model.optimize()

        return C1PResult(status = True if model.Status == 2 else False, time=0)


def findCRPoints(A: ndarray, C: List[str], V: List[str], env):  # voter radius = 0
    indexPairs = list(product(range(len(C)), range(len(V))))

    if env is None:
        env = createGPEnv()

    with gp.Model(env=env) as model:
        # Xa variable - positon of agent
        positions = {
            "x" + agent: model.addVar(vtype=gp.GRB.CONTINUOUS, name="x" + agent) for agent in C + V
        }

        # Ra variable - radius of agent
        radiuses = {
            "r" + agent: model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="r" + agent) for agent in C + V
        }

        Z1Vars = {
            C[c] + V[v]: model.addVar(vtype=gp.GRB.BINARY, name="Z1" + C[c] + V[v]) for c, v in indexPairs
        }

        Z2Vars = {
            C[c] + V[v]: model.addVar(vtype=gp.GRB.BINARY, name="Z2" + C[c] + V[v]) for c, v in indexPairs
        }

        # c,v - agent ID (name), cI, vI - agent index
        for v in V:
            model.addConstr(radiuses['r' + v] == 0, "CR-" + v)
        for (c, v), (cI, vI) in zip(product(C, V), product(range(len(C)), range(len(V)))):
            if A[vI, cI] == 1:
                model.addConstr(positions['x' + c] - positions['x' + v] <= radiuses['r' + c] + radiuses['r' + v],
                                "VCR-pos-" + c + v)
                model.addConstr(-(positions['x' + c] - positions['x' + v]) <= radiuses['r' + c] + radiuses['r' + v],
                                "VCR-neg-" + c + v)
            else:
                model.addConstr(
                    (Z1Vars[c + v] == 1) >> ((positions['x' + c] - positions['x' + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1)
                )
                model.addConstr(
                    (Z2Vars[c + v] == 1) >> ((-positions['x' + c] + positions['x' + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1)
                )

                model.addConstr(Z1Vars[c + v] + Z2Vars[c + v] >= 1)

        model.setParam('OutputFlag', False)
        model.optimize()

        if model.Status == 2:
            return True, {v.varName: v.X for v in model.getVars() if 'r' in v.varName or 'x' in v.varName}
        else:
            return False, {"status": model.status}

