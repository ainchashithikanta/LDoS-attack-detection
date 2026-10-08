import numpy as np

# E = - summation of(mi*log(mi))
# mi = wi**2
def calculate_entropy(W):
    M= np.abs(W) ** 2

    total = np.sum(M)

    M = M/total   
    # weighted mi

    entropy = -np.sum(M*np.log(M))

    return entropy