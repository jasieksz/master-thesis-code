from pulp import *
import itertools
from typing import List

def detectorMockDist(A, V: List[str], C: List[str],):
    M = 100
    indexPairs = list(itertools.product(list(range(len(C))), list(range(len(V)))))

    prob = LpProblem("vcr", LpMinimize)

    # Dcv variable - distance between c and v
    distanceVars = {
        C[c]+V[v] : LpVariable("D" + C[c]+V[v],None, None, LpContinuous) for c,v in indexPairs
    }  

    # Dcv variable - distance between c and v
    distanceZVars = {
        C[c]+V[v] : LpVariable("ZD" + C[c]+V[v],0, 1, LpBinary) for c,v in indexPairs
    }    

    # Xa variable - positon of agent
    positionVars = {
        "x" + agent : LpVariable("x" + agent,None,None,LpContinuous) for agent in C + V
    }

    # Ra variable - radius of agent
    radiusVars = {
        "r" + agent : LpVariable("r" + agent,0,None,LpContinuous) for agent in C + V
    }

    # Objective function, alternatively 0
    # prob += lpSum(list(radiusVars.values())) , "fun"
    prob += 0 , "fun"

    # c,v - agent ID (name), cI, vI - agent index
    for (c,v),(cI, vI) in zip(itertools.product(C, V),itertools.product(range(len(C)), range(len(V)))):
        if (A[vI, cI] == 1):
            prob += distanceVars[c+v] <= radiusVars['r'+c] + radiusVars['r'+v], "VCR" + c + v
            prob += (distanceVars[c+v] == positionVars['x'+c] - positionVars['x'+v]), "D"+c+v +"-pos"
            prob += distanceVars[c+v] == -(positionVars['x'+c] - positionVars['x'+v]), "D"+c+v +"-neg"
        else:
            prob += distanceVars[c+v] >= radiusVars['r'+c] + radiusVars['r'+v] + 1, "notVCR" + c + v
            prob += (distanceVars[c+v] == positionVars['x'+c] - positionVars['x'+v]), "D"+c+v +"-pos"
            prob += (distanceVars[c+v] == -(positionVars['x'+c] - positionVars['x'+v])), "D"+c+v +"-neg"

    prob.solve()
    ilpResult = {var.name : var.varValue for var in prob.variables()}
    return prob, ilpResult

def detectorPosNeg(A, V: List[str], C: List[str]):
    M = 100
    indexPairs = list(itertools.product(list(range(len(C))), list(range(len(V)))))

    prob = LpProblem("vcr", LpMinimize)

    # Xa variable - positon of agent
    positions = {
        "x" + agent : LpVariable("x" + agent,None,None,LpContinuous) for agent in C + V
    }

    # Ra variable - radius of agent
    radiuses = {
        "r" + agent : LpVariable("r" + agent,0,None,LpContinuous) for agent in C + V
    }

    Z1Vars = {
        C[c]+V[v] : LpVariable("Z1" + C[c]+V[v],0, 1, LpBinary) for c,v in indexPairs
    } 
    
    Z2Vars = {
        C[c]+V[v] : LpVariable("Z2" + C[c]+V[v],0, 1, LpBinary) for c,v in indexPairs
    } 

    # Objective function, 0
    prob += 0 , "fun"

    # c,v - agent ID (name), cI, vI - agent index
    for (c,v),(cI, vI) in zip(itertools.product(C, V),itertools.product(range(len(C)), range(len(V)))):
        if (A[vI, cI] == 1):
            prob += (positions['x'+c] - positions['x'+v]) <= radiuses['r'+c] + radiuses['r'+v], "VCR-pos-" + c + v 
            prob += -(positions['x'+c] - positions['x'+v]) <= radiuses['r'+c] + radiuses['r'+v], "VCR-neg-" + c + v
        else:
            prob += positions['x'+c] - positions['x'+v] + M*(1 - Z1Vars[c+v]) >= radiuses['r'+c] + radiuses['r'+v] + 1, "notVCR-pos-" + c + v
            prob += -(positions['x'+c] - positions['x'+v]) + M*(1 - Z2Vars[c+v]) >= radiuses['r'+c] + radiuses['r'+v] + 1, "notVCR-neg-" + c + v
            prob += Z1Vars[c+v] + Z2Vars[c+v] >= 1
            
    prob.solve()
    ilpResult = {var.name : var.varValue for var in prob.variables()}
    return prob, ilpResult