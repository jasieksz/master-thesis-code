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
