import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
df = pd.read_csv("B0005_thermal_events.csv")

df = df.sort_values(["cycle", "time"])


cycle_data = df.groupby("cycle").agg(
    events=("thermal_event", "sum"),
    peak_temperature=("temperature", "max")
).reset_index()


# =========================================================
# GRAPH 1: THERMAL EVENTS VS BATTERY CYCLE
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(
    cycle_data["cycle"],
    cycle_data["events"],
    marker="o",
    markersize=3,
    linewidth=1
)

plt.xlabel("Discharge Cycle")
plt.ylabel("Thermal Event Readings")
plt.title("Thermal Event Frequency Across Battery Life")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("thermal_events_vs_cycle.png", dpi=300)
plt.show()


# =========================================================
# POISSON MODEL
# =========================================================

counts = cycle_data["events"].to_numpy()

lam = np.mean(counts)

print("\nPOISSON ANALYSIS")
print("----------------")
print(f"Number of cycles = {len(counts)}")
print(f"Mean λ = {lam:.4f}")
print(f"Variance = {np.var(counts, ddof=1):.4f}")
print(
    f"Dispersion index = "
    f"{np.var(counts, ddof=1) / lam:.4f}"
)


# =========================================================
# GRAPH 2: OBSERVED VS POISSON
# =========================================================

max_k = int(counts.max())

k_values = np.arange(0, max_k + 1)

observed = np.array([
    np.sum(counts == k)
    for k in k_values
])

# Poisson probabilities
poisson_probability = np.array([
    math.exp(-lam) * lam**k / math.factorial(k)
    for k in k_values
])

# Convert probabilities to expected number of cycles
expected = poisson_probability * len(counts)


plt.figure(figsize=(11, 5))

width = 0.4

plt.bar(
    k_values - width / 2,
    observed,
    width=width,
    label="Observed"
)

plt.bar(
    k_values + width / 2,
    expected,
    width=width,
    label="Poisson Expected"
)

plt.xlabel("Thermal Event Readings per Cycle")
plt.ylabel("Number of Cycles")

plt.title(
    f"Observed vs Poisson Distribution "
    f"(λ = {lam:.2f})"
)

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig("poisson_observed_vs_expected.png", dpi=300)
plt.show()


# =========================================================
# EXPONENTIAL DATA
# =========================================================

# Find the first thermal event in each cycle
first_events = (
    df[df["thermal_event"] == 1]
    .groupby("cycle")["time"]
    .min()
)

waiting_times = first_events.to_numpy()

lambda_exp = 1 / np.mean(waiting_times)

print("\nEXPONENTIAL ANALYSIS")
print("--------------------")
print(f"Cycles with an event = {len(waiting_times)}")
print(f"Mean first-event time = {np.mean(waiting_times):.2f} seconds")
print(f"λ = {lambda_exp:.6f} per second")


# =========================================================
# GRAPH 3: FIRST EVENT WAITING TIME
# =========================================================

plt.figure(figsize=(10, 5))

plt.hist(
    waiting_times,
    bins=15,
    density=True,
    alpha=0.6,
    label="Observed"
)


# Exponential curve
t = np.linspace(
    0,
    waiting_times.max(),
    300
)

exponential_pdf = (
    lambda_exp *
    np.exp(-lambda_exp * t)
)

plt.plot(
    t,
    exponential_pdf,
    linewidth=2,
    label="Exponential Model"
)

plt.xlabel("Time to First Thermal Event (seconds)")
plt.ylabel("Probability Density")

plt.title(
    "Observed First-Event Waiting Times vs Exponential Model"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("exponential_waiting_time.png", dpi=300)
plt.show()