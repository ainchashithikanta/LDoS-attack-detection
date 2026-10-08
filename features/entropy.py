import numpy as np

def calculate_entropy(W):
    M= np.abs(W) ** 2

    total = np.sum(M)

    M = M/total

    entropy = -np.sum(M*np.log(M))

    return entropy