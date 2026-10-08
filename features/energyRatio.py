def calculate_energy_ratio(energy):
    row, column = energy.shape

    low_frequency_energy = 0
    total_energy = 0

    low_frequency_limit = row // 5

    for i in range(row):
        for j in range(column):
            total_energy += energy[i, j]

            if i < low_frequency_limit:
                low_frequency_energy += energy[i, j]

    return low_frequency_energy / total_energy