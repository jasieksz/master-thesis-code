#%%
import gurobipy as gp
from itertools import product

#%%
def detectVCRProperty(A, V: List[str], C: List[str]):
    indexPairs = list(product(range(len(C)), range(len(V))))

    with gp.Env(empty=True) as env:
        # env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as model:
            
