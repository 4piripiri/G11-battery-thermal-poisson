import math
import pandas as pd

# Load battery data
df = pd.read_csv("B0005_thermal_events.csv")

# Use the same threshold as the main project
THRESHOLD = 40.0

# Mark thermal events
df["thermal_event"] = (df["temperature"] > THRESHOLD).astype(int)

# Get the first thermal-event time for each cycle
first_events = (
    df[df["thermal_event"] == 1]
    .groupby("cycle")["time"]
    .min()
)

# Remove cycles where no thermal event occurred
first_events = first_events.dropna()

# Calculate average waiting time
mean_waiting_time = first_events.mean()

# Exponential rate parameter
lam = 1 / mean_waiting_time

print("Number of cycles with thermal events:", len(first_events))
print("Mean waiting time:", mean_waiting_time, "seconds")
print("Lambda:", lam, "per second")

# Example: probability that first thermal event occurs after 1000 seconds
t = 1000

p_greater = math.exp(-lam * t)
p_less = 1 - p_greater

print("P(T > 1000 seconds):", p_greater)
print("P(T <= 1000 seconds):", p_less)

# Expected waiting time according to exponential model
expected_waiting_time = 1 / lam

print("Expected waiting time:", expected_waiting_time, "seconds")