import numpy as np

def calculate_correlation(energy):
    row,columns=energy.shape
    contrast=0
    for i in range(row):
        for j in range(columns):
            contrast+= (i-j) ** 2 * energy[i,j]

    return contrast