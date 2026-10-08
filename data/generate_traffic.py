import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#take ldos parameters from the user

T=float(input("Enter attack period T(seconds): "))
L=float(input("Enter attack duration L(seconds): "))
R=float(input("Enter attack rate R(Mbps): "))

#validate the inputs

if T <= 0:
    print("Error: Attack period T must be greater than 0.")
    exit()

if L <= 0:
    print("Error: Attack duration L must be greater than 0.")
    exit()

if R <= 0:
    print("Error: Attack rate R must be greater than 0.")
    exit()

if L >= T:
    print("Error: Attack duration L must be less than attack period T.")
    exit()


#general settings
duration = 200 #seconds
sampling_rate = 10 #samples per second

number_of_samples = duration * sampling_rate

time = np.arange(number_of_samples)/sampling_rate


#generate traffic data
np.random.seed(42) 

normal_traffic = 10+np.random.normal(0, 0.5, size=number_of_samples)  # Normal traffic around 10 Mbps with some noise


#generate ldos attack traffic

ldos_traffic = normal_traffic.copy()

for i in range(number_of_samples):

    current_time = time[i]
    position_in_period = current_time % T

    if position_in_period < L:
        ldos_traffic[i] += R  # Add attack rate during the attack duration


#create normal traffic DataFrame

normal_traffic_df = pd.DataFrame({
    'Time (s)': time,
    'Traffic (Mbps)': normal_traffic,
    'label':0
})

#create ldos traffic DataFrame

ldos_traffic_df = pd.DataFrame({
    'Time (s)': time,
    'Traffic (Mbps)': ldos_traffic,
    'label':1
})

#save the data to CSV files
normal_traffic_df.to_csv('normal_traffic.csv', index=False)
ldos_traffic_df.to_csv('ldos_traffic.csv', index=False)

print("Traffic data generated and saved to 'data/normal_traffic.csv' and 'data/ldos_traffic.csv'.")

print("\nparameters used for the ldos attack:")
print(f"Attack period (T): {T} seconds")
print(f"Attack duration (L): {L} seconds")
print(f"Attack rate (R): {R} Mbps")

