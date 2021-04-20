#%%
from definitions import Profile, Candidate, Voter
import numpy as np
from mavUtils import getVCRProfileInCROrder, getVCRProfileInVROrder

def VCRNCOP_44():
    A = np.load("resources/output/4C4V/NCOP-profiles/ncop-44-0.npy")
    return list(map(Profile.fromNumpy, A))

def VCRNCOP_55_1():
    A = np.load("resources/output/5C5V/NCOP-profiles/ncop-55-1.npy")
    return list(map(Profile.fromNumpy, A))

def VCRNCOP_55_2():
    A = np.load("resources/output/5C5V/NCOP-profiles/ncop-55-2.npy")
    return list(map(Profile.fromNumpy, A))

def VCRNCOP_55_3():
    A = np.load("resources/output/5C5V/NCOP-profiles/ncop-55-3.npy")
    return list(map(Profile.fromNumpy, A))

def VCRNCOP_66():
    A = np.load("resources/output/6C6V/NCOP-profiles/ncop-66-0.npy")
    return list(map(Profile.fromNumpy, A))

def VCRNCOP_1010():
    A = np.load("resources/output/10C10V/NCOP-profiles/ncop-1010-0.npy")
    return list(map(Profile.fromNumpy, A))

def VCR_1515_01k():
    A = np.load("resources/output/15C15V/numpy/1515-0.npy")
    return list(map(Profile.fromNumpy, A))


# def CR_66_0():
#     A = np.load("resources/output/6C6V/CR-profiles/cr-66-0.npy")
#     return list(map(getVCRProfileInCROrder,map(Profile.fromNumpy, A)))

# def VR_66_0():
#     A = np.load("resources/output/6C6V/VR-profiles/vr-66-0.npy")
#     return list(map(getVCRProfileInVROrder,map(Profile.fromNumpy, A)))
