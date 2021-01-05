import multiprocessing
from numpy import array, copy, ndarray, save, load
from itertools import product, permutations
from typing import List
import time


def parallelProfileGeneration(cpu: int, candidatesNumber: int, voterNumber: int):
    with multiprocessing.Pool(cpu) as pool:
        singleVotes = list(pool.map(list, product([0, 1], repeat=candidatesNumber)))
        profiles = list(pool.map(list, product(singleVotes, repeat=voterNumber)))
        profilesAsNp = list(pool.map(array, profiles))
    return array(profilesAsNp)


def run(C,V):
    P55 = parallelProfileGeneration(16,C,V)
    path = "resources/profiles/all-5C-5V/P55-{}.npy"
    w = P55.shape[0] // 32

    for i in range(32):
        with open(path.format(i), 'wb') as f:
            save(f, P55[i*w:(i+1)*w])


def lod(C,V):
    loadPath = "resources/profiles/all-5C-5V/P55-{}.npy"
    savePath = "resources/profiles/all-5C-5V/P55-{}-{}.npy"

    for i in range(32):
        with open(loadPath.format(i), 'rb') as fr:
            P55_i32 = load(fr)
        for j in range(4):
            w = 262144
            with open(savePath.format(i,j), 'wb') as fs:
                if (j == 3):
                    save(fs, P55[j*w:(j+1)*w])
                save(fs, P55[j*w:(j+1)*w])


    print(sum)


# startTime = time.time()
# lod(5,5)
# print("Finished in ", time.time() - startTime)

print(load("resources/profiles/all-5C-5V/P55-{}.npy".format(7)).shape)