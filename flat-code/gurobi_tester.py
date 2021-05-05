import gurobipy as gp
from itertools import product
from typing import List
from numpy import ndarray


# %%
def createGPEnv():
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    return env


if __name__ == "__main__":
    print("START")
    env = createGPEnv()
    print("DONE")
