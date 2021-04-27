#%%
from typing import List, Tuple, Dict
from numpy import ndarray
from vcrDetectionAlt import findVRPoints, detectVCRProperty
from collections import namedtuple
import numpy as np
from definitions import Profile
from vcrDomain import isVCR

#%%
IdPosition = namedtuple('IdPosition', ['id', 'x'])

def getVROrder(votersIds: List[str], votersPoints: Dict[str,float]) -> List[int]:
    idsPoints = [IdPosition(i,votersPoints['x'+vId]) for i,vId in enumerate(votersIds)]
    idsPoints.sort(key=lambda id_x: id_x[1])
    return [idx.id for idx in idsPoints]

def shuffleCols(array:np.ndarray, order:list) -> np.ndarray:
    A = np.array(array).transpose()
    A[list(range(A.shape[0]))] = A[order]
    return A.transpose()

def getFullProfileVRFromVCR(P:Profile, gurobiEnv=None) -> Tuple[bool,Profile]:
    V,C = P.A.shape
    vIds = ['v'+str(i) for i in range(V)]
    cIds = ['c'+str(i) for i in range(C)]
    foundSolution, positionDict = findVRPoints(P.A, cIds, vIds, gurobiEnv)
    if (foundSolution == False):
        return (False,None)
    voterOrder = getVROrder(cIds, positionDict)
    vrA = shuffleCols(P.A, voterOrder)
    pTmp = Profile.fromILPRes(vrA, positionDict, cIds, vIds)
    pResult = getFullProfileVRFromVCRHelper(pTmp, gurobiEnv)
    if isVCR(pResult):
        return (True,pResult)
    else:
        print("BOOOOM")
        (False, None)

def getFullProfileVRFromVCRHelper(P:Profile, gurobiEnv=None) -> Profile:
    V,C = P.A.shape
    vIds = ['v'+str(i) for i in range(V)]
    cIds = ['c'+str(i) for i in range(C)]
    foundSolution, positionDict = findVRPoints(P.A, cIds, vIds, gurobiEnv)
    if (foundSolution == False):
        print("BOOOOM 2")
        return None
    return Profile.fromILPRes(P.A, positionDict, cIds, vIds)   
