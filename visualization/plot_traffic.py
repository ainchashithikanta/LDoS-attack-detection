import pandas as pd
import matplotlib.pyplot as plt


# Load the data

normal_traffic = pd.read_csv("../data/normal_traffic.csv")
ldos_traffic = pd.read_csv("../data/ldos_traffic.csv")


# Create one figure with two subplots

fig, axes = plt.subplots(2, 1, figsize=(12, 8))


# Normal traffic

axes[0].plot(
    normal_traffic["Time (s)"],
    normal_traffic["Traffic (Mbps)"]
)

axes[0].set_title("Normal Network Traffic")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Traffic (Mbps)")
axes[0].grid()


# LDoS traffic

axes[1].plot(
    ldos_traffic["Time (s)"],
    ldos_traffic["Traffic (Mbps)"]
)

axes[1].set_title("LDoS Network Traffic")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Traffic (Mbps)")
axes[1].grid()


# Automatically adjust spacing

plt.tight_layout()

plt.show()