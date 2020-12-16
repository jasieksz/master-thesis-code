import gurobipy as gp
from itertools import product
from typing import List
from numpy import ndarray


# %%
def detectVCRProperty(A: ndarray, C: List[str], V: List[str]):
    indexPairs = list(product(range(len(C)), range(len(V))))

    with gp.Model() as model:
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
        M = 100
        for (c, v), (cI, vI) in zip(product(C, V), product(range(len(C)), range(len(V)))):
            if A[vI, cI] == 1:
                model.addConstr(positions['x' + c] - positions['x' + v] <= radiuses['r' + c] + radiuses['r' + v], "VCR-pos-" + c + v)
                model.addConstr(-(positions['x' + c] - positions['x' + v]) <= radiuses['r' + c] + radiuses['r' + v], "VCR-neg-" + c + v)
            else:
                model.addConstr(positions['x' + c] - positions['x' + v] + M * (1 - Z1Vars[c + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1, "notVCR-pos-" + c + v)
                model.addConstr(-(positions['x' + c] - positions['x' + v]) + M * (1 - Z2Vars[c + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1, "notVCR-neg-" + c + v)
                model.addConstr(Z1Vars[c + v] + Z2Vars[c + v] >= 1)


        model.setParam('OutputFlag', False)
        # model.setParam('FeasibilityTol', GUROBI_ACCURACY)
        # model.setParam('PoolSearchMode', 0)

        model.optimize()
        # print(model.Status)
        if model.Status == 2:
            return model.Status, {v.varName: v.X for v in model.getVars() if 'r' in v.varName or 'x' in v.varName}
        else:
            return model.Status, {}

# %%
def detectVRProperty(A: ndarray, C: List[str], V: List[str]): # CR = 0
    indexPairs = list(product(range(len(C)), range(len(V))))

    with gp.Model() as model:
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
        M = 100
        for c in C:
            model.addConstr(radiuses['r' + c] == 0, "VR-"+c)
        for (c, v), (cI, vI) in zip(product(C, V), product(range(len(C)), range(len(V)))):
            if A[vI, cI] == 1:
                model.addConstr(positions['x' + c] - positions['x' + v] <= radiuses['r' + c] + radiuses['r' + v], "VCR-pos-" + c + v)
                model.addConstr(-(positions['x' + c] - positions['x' + v]) <= radiuses['r' + c] + radiuses['r' + v], "VCR-neg-" + c + v)
            else:
                model.addConstr(positions['x' + c] - positions['x' + v] + M * (1 - Z1Vars[c + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1, "notVCR-pos-" + c + v)
                model.addConstr(-(positions['x' + c] - positions['x' + v]) + M * (1 - Z2Vars[c + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1, "notVCR-neg-" + c + v)
                model.addConstr(Z1Vars[c + v] + Z2Vars[c + v] >= 1)


        model.setParam('OutputFlag', False)
        model.optimize()

        return True if model.Status == 2 else False


# %%
def detectCRProperty(A: ndarray, C: List[str], V: List[str]): # voter radius = 0
    indexPairs = list(product(range(len(C)), range(len(V))))

    with gp.Model() as model:
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
        M = 100
        for v in V:
            model.addConstr(radiuses['r' + v] == 0, "CR-"+v)
        for (c, v), (cI, vI) in zip(product(C, V), product(range(len(C)), range(len(V)))):
            if A[vI, cI] == 1:
                model.addConstr(positions['x' + c] - positions['x' + v] <= radiuses['r' + c] + radiuses['r' + v], "VCR-pos-" + c + v)
                model.addConstr(-(positions['x' + c] - positions['x' + v]) <= radiuses['r' + c] + radiuses['r' + v], "VCR-neg-" + c + v)
            else:
                model.addConstr(positions['x' + c] - positions['x' + v] + M * (1 - Z1Vars[c + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1, "notVCR-pos-" + c + v)
                model.addConstr(-(positions['x' + c] - positions['x' + v]) + M * (1 - Z2Vars[c + v]) >= radiuses['r' + c] + radiuses['r' + v] + 1, "notVCR-neg-" + c + v)
                model.addConstr(Z1Vars[c + v] + Z2Vars[c + v] >= 1)


        model.setParam('OutputFlag', False)
        model.optimize()

        return True if model.Status == 2 else False

