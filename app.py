import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Find fswt.py
sys.path.append("features")
from fswt import fswt

# Load traffic data
normal_data = pd.read_csv("data/normal_traffic.csv")
ldos_data = pd.read_csv("data/ldos_traffic.csv")
print("Traffic data loaded.")

# Extract signals
normal_signal = normal_data["Traffic (Mbps)"].values
ldos_signal = ldos_data["Traffic (Mbps)"].values

# Sampling rate
sampling_rate = 10
print("Sampling rate:", sampling_rate, "samples/second")

# Remove DC component
normal_signal = normal_signal - np.mean(normal_signal)
ldos_signal = ldos_signal - np.mean(ldos_signal)
print("DC component removed.")

# Calculate FSWT
print()
print("Calculating FSWT for normal traffic...")
normal_fswt = fswt(normal_signal, sampling_rate)
print("Normal FSWT completed.")

print()
print("Calculating FSWT for LDoS traffic...")
ldos_fswt = fswt(ldos_signal, sampling_rate)
print("LDoS FSWT completed.")

# Calculate energy
normal_energy = np.abs(normal_fswt) ** 2
ldos_energy = np.abs(ldos_fswt) ** 2
print()
print("Energy matrices created.")

# Frequency axis
N = len(ldos_signal)
frequencies = np.fft.fftfreq(N, d=1 / sampling_rate)

# Positive frequencies
positive = frequencies >= 0
frequencies_positive = frequencies[positive]
normal_energy_positive = normal_energy[positive, :]
ldos_energy_positive = ldos_energy[positive, :]

# Attack parameters
T = 15.0
attack_frequency = 1 / T

print()
print("LDoS attack period:", T, "seconds")
print("Expected attack frequency:", attack_frequency, "Hz")

# Find strongest frequency in LDoS
average_energy = np.mean(ldos_energy_positive, axis=1)

# Ignore DC
average_energy[0] = 0

peak_index = np.argmax(average_energy)
peak_frequency = frequencies_positive[peak_index]

print("Strongest FSWT frequency:", peak_frequency, "Hz")

# Find closest frequency bin
closest_index = np.argmin(np.abs(frequencies_positive - attack_frequency))
closest_frequency = frequencies_positive[closest_index]

print("Closest frequency bin:", closest_frequency, "Hz")

# Check harmonics
print()
print("Harmonic analysis:")

for harmonic in range(1, 6):
    expected_frequency = harmonic / T
    index = np.argmin(np.abs(frequencies_positive - expected_frequency))
    actual_frequency = frequencies_positive[index]
    energy = average_energy[index]

    print("Harmonic:", harmonic, "| Expected:", expected_frequency, "Hz", "| Bin:", actual_frequency, "Hz", "| Energy:", energy)
# Plot everything
fig, axes = plt.subplots(4, 1, figsize=(12, 14))

# Plot 1: Normal traffic
axes[0].plot(normal_data["Time (s)"], normal_data["Traffic (Mbps)"])
axes[0].set_title("Normal Network Traffic")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Traffic (Mbps)")
axes[0].grid()

# Plot 2: LDoS traffic
axes[1].plot(ldos_data["Time (s)"], ldos_data["Traffic (Mbps)"])
axes[1].set_title("LDoS Network Traffic")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Traffic (Mbps)")
axes[1].grid()

# Plot 3: Normal FSWT energy
axes[2].imshow(normal_energy_positive.T, aspect="auto", origin="lower", extent=[frequencies_positive[0], frequencies_positive[-1], 0, len(normal_signal) / sampling_rate])
axes[2].set_title("FSWT Energy - Normal Traffic")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Time (seconds)")

# Plot 4: LDoS FSWT energy
axes[3].imshow(ldos_energy_positive.T, aspect="auto", origin="lower", extent=[frequencies_positive[0], frequencies_positive[-1], 0, len(ldos_signal) / sampling_rate])
axes[3].set_title("FSWT Energy - LDoS Traffic")
axes[3].set_xlabel("Frequency (Hz)")
axes[3].set_ylabel("Time (seconds)")

# Adjust spacing
plt.tight_layout()

# Show plots
plt.show()