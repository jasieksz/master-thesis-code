from time import time
from pickle import dump, load


def timed(function):
    startTime = time()
    result = function()
    endTime = time()
    return result, endTime - startTime


def savePickle(path: str, item):
    with open(path, 'wb') as f:
        dump(item, f)


def loadPickle(path: str):
    with open(path, 'rb') as f:
        item = load(f)
    return item

def getNumpyColumns(C, V):
    res = []
    res.append("C")
    res.append("V")
    for i in range(C):
        res.append("C"+str(i)+"x")
        res.append("C"+str(i)+"r")
    for i in range(V):
        res.append("V"+str(i)+"x")
        res.append("V"+str(i)+"r")
    for v in range(V):
        for c in range(C):
            res.append("A"+str(v)+"_"+str(c))
    return res