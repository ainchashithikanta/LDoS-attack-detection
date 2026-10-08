import numpy as np

def calculate_correlation(energy):
    row, column = energy.shape

    total_energy = np.sum(energy)

    mu1 = 0
    mu2 = 0

    for i in range(row):
        for j in range(column):
            mu1 += i * energy[i, j]
            mu2 += j * energy[i, j]

    mu1 = mu1 / total_energy
    mu2 = mu2 / total_energy

    sigma1_squared = 0
    sigma2_squared = 0

    for i in range(row):
        for j in range(column):
            sigma1_squared += (i - mu1) ** 2 * energy[i, j]
            sigma2_squared += (j - mu2) ** 2 * energy[i, j]

    sigma1_squared = sigma1_squared / total_energy
    sigma2_squared = sigma2_squared / total_energy

    numerator = 0

    for i in range(row):
        for j in range(column):
            numerator += i * j * energy[i, j]

    numerator = numerator / total_energy
    numerator = numerator - mu1 * mu2

    correlation = numerator / (sigma1_squared * sigma2_squared)

    return correlation