import numpy as np
import sys

def hammingDistance(a: np.ndarray, b: np.ndarray) -> int:
    return (a != b).nonzero()[0].shape[0]

if __name__ == "__main__":    
    vectorSize = int(sys.argv[1])
    a = np.array([int(e) for e in sys.argv[2:2+vectorSize]])
    b = np.array([int(e) for e in sys.argv[2+vectorSize:2+vectorSize+vectorSize]])
    print("HD = {}\n{}\n{}".format(hammingDistance(a,b), a, b))

