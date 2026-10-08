import numpy as np

def calculate_contrast(energy):
    row, column = energy.shape
    contrast = 0

    for i in range(row):
        for j in range(column):
            contrast += (i - j) ** 2 * energy[i, j]

    return contrast