from numpy import ndarray
from code.profiles.generation import getProfilePermutations

def consecutiveOnes1D(A: ndarray) -> bool:
    ones: bool = False
    afterOnes: bool = False
    for i in range(len(A)):
        if A[i] == 1 and afterOnes == False:
            ones = True
        elif A[i] == 0 and ones == True:
            afterOnes = True
        elif A[i] == 1 and afterOnes == True:
            return False
    return True

def consecutiveOnesCR(approvalSet: ndarray) -> bool:
    return all([consecutiveOnes1D(approvalSet[:, i]) for i in range(approvalSet.shape[1])])

def consecutiveOnesVR(approvalSet: ndarray) -> bool:
    return all([consecutiveOnes1D(approvalSet[i, :]) for i in range(approvalSet.shape[0])])

def consecutiveOnes2D(approvalSet: ndarray) -> bool:
    return consecutiveOnesCR(approvalSet) or consecutiveOnesVR(approvalSet)

def totalConsecutiveOnes2D(approvalSet: ndarray) -> bool:
    return any([consecutiveOnes2D(approvalSet=a) for a in getProfilePermutations(approvalSet)])