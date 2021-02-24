#%%
from typing import List, Tuple, Dict
from numpy import ndarray
from vcrDetectionAlt import findCRPoints
from collections import namedtuple
import numpy as np

#%%
IdPosition = namedtuple('IdPosition', ['id', 'x'])

def getCROrder(votersIds: List[str], votersPoints: Dict[str,float]) -> List[int]:
    idsPoints = [IdPosition(i,votersPoints['x'+vId]) for i,vId in enumerate(votersIds)]
    idsPoints.sort(key=lambda id_x: id_x[1])
    return [idx.id for idx in idsPoints]

def shuffleToCR(A:np.ndarray, voterOrder:List[int]) -> np.ndarray:
    A_CR = np.array(A)
    A_CR[list(range(A.shape[0]))] = A_CR[voterOrder]
    return A_CR

def getProfileCRFromVCR(A:np.ndarray, gurobiEnv=None) -> Tuple[bool,np.ndarray]:
    V,C = A.shape
    vIds = ['v'+str(i) for i in range(V)]
    cIds = ['c'+str(i) for i in range(C)]
    foundSolution, positionDict = findCRPoints(A, cIds, vIds, gurobiEnv)
    if (foundSolution == False):
        return (False,A)
    voterOrder = getCROrder(vIds, positionDict)
    return (True, shuffleToCR(A, voterOrder))