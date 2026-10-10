import math
import pandas as pd

df = pd.read_csv("B0005_thermal_events.csv")
df = df[(df["cycle"] >= 400) & (df["cycle"] <= 614)]

counts = df.groupby("cycle")["thermal_event"].sum()

high = []
for c in counts.index:
    if counts[c] >= 18:
        high.append(c)

# gaps between high cycles
gaps = []
for i in range(1, len(high)):
    gaps.append(high[i] - high[i-1])

mean_gap = sum(gaps) / len(gaps)
lam = 1 / mean_gap

print("gaps:", gaps)
print("mean gap:", mean_gap)
print("lambda:", lam)

t = 10
p_greater = math.exp(-lam * t)
p_less = 1 - p_greater

print("P(T > 10):", p_greater)
print("P(T <= 10):", p_less)
