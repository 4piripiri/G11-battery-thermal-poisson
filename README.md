# ⚡ Battery Thermal Anomaly Risk Screening Dashboard

An applied statistical engineering dashboard built with **Streamlit**, **NumPy**, **SciPy**, **Pandas**, and **Plotly**. The application integrates empirical Lithium-ion battery telemetry (NASA Li-ion aging dataset), Poisson counting processes, Exponential waiting times, and Monte Carlo stochastic simulations to support explainable Battery Management System (BMS) risk decisions.

---

## 🎯 Engineering & Statistical Problem Statement

In Battery Energy Storage Systems (BESS) and Electric Vehicles (EVs), thermal management systems monitor thermistors across hundreds of cells. A common statistical pitfall in battery telemetry is **autocorrelation / false alarms caused by naive counting**:
- When a cell overheats during discharge, thermistors scan high temperatures repeatedly (e.g., every 18 seconds).
- The supplied NASA B0005 dataset contains **1,206 flagged telemetry readings** (`thermal_event == 1`).
- Naively treating every flagged reading as an independent event yields an arrival rate of $\frac{1,206}{146.0\text{ h}} = 8.26\text{ events/hour}$—an overestimation of over **1400%**.
- In this application, an **episode** is correctly defined as a run of consecutive flagged readings within a discharge cycle, counted at its **initial onset** ($0 \to 1$ transition). This identifies **86 discrete thermal episodes**, yielding a statistically sound arrival rate of:
  $$\hat{\lambda}_{\text{obs}} = \frac{86}{146.04\text{ hours}} = 0.589\text{ episodes/hour} \quad (\text{Exact 95% Poisson CI: } [0.471, 0.727]\text{/hr})$$

---

## 🚀 Quickstart & Setup Instructions

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10–3.14)
- Pip package manager

### 2. Installation
Open PowerShell or your terminal in the project directory:

```bash
cd c:\Users\Admin\Desktop\battery_thermal_sim
pip install -r requirements.txt
```

### 3. Running the Application
Launch via Streamlit:

```bash
python -m streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 🏗️ Architecture & Core Modules

The application is structured into modular Python components:

| File | Purpose |
| :--- | :--- |
| [`app.py`](file:///c:/Users/Admin/Desktop/battery_thermal_sim/app.py) | Main interactive Streamlit application with 5 dedicated operational tabs, sidebar controls, dynamic badges, KPI cards, and CSV exports. |
| [`core.py`](file:///c:/Users/Admin/Desktop/battery_thermal_sim/core.py) | Computational core with CSV validation, onset extraction, Garwood exact Poisson confidence intervals, cycle telemetry aggregations, and Monte Carlo generation. |
| [`diagnostics.py`](file:///c:/Users/Admin/Desktop/battery_thermal_sim/diagnostics.py) | Plotly goodness-of-fit charts evaluating Poisson dispersion, Exponential gaps, operating history trends, and aging-induced non-stationarity. |
| [`B0005_thermal_events.csv`](file:///c:/Users/Admin/Desktop/battery_thermal_sim/B0005_thermal_events.csv) | Bundled NASA Lithium-ion cell B0005 discharge cycling dataset (50,285 telemetry rows across 168 discharge cycles). |
| [`requirements.txt`](file:///c:/Users/Admin/Desktop/battery_thermal_sim/requirements.txt) | Pinned application dependencies (`streamlit`, `numpy`, `scipy`, `pandas`, `plotly`). |

---

## 🔬 Model Diagnostics & Goodness-of-Fit

The dashboard investigates the validity of Poisson and Exponential assumptions against real telemetry:

1. **Cycle-Level Counts & Under-Dispersion:**
   - In a theoretical Poisson process, the variance-to-mean dispersion index is $D = \frac{\text{Var}(N)}{\mu} = 1.0$.
   - On the NASA dataset, observed per-cycle counts have mean $\mu = 0.512$ and variance $\sigma^2 = 0.251$, resulting in $D = 0.491$.
   - **Reason:** Each discharge cycle has at most 1 episode onset near the end of discharge ($k=0: 82\text{ cycles}, k=1: 86\text{ cycles}, k \ge 2: 0\text{ cycles}$). The data behaves as a Bernoulli trial per cycle rather than an unconstrained Poisson variable.
2. **Waiting Times & Violation of Memorylessness:**
   - Observed inter-arrival gaps have mean $1.18\text{ h}$ and standard deviation $3.16\text{ h}$ (Coefficient of Variation $CV = \frac{\sigma}{\mu} = 2.68$).
   - For a true Exponential distribution, $CV = 1.0$. Gaps cluster heavily near ~0.85 hours in late life, while remaining very long in early life.
3. **Severe Non-Stationarity (Aging-Induced Rate Escalation):**
   - Early cycles (1–50) exhibit an anomaly rate of only **2%**.
   - Middle cycles (50–100) exhibit a rate of **34%**.
   - Late cycles (100–168) exhibit a rate of **100%** (anomalies occur in every single cycle).
   - This proves that $\lambda(t)$ is not constant over battery lifetime. Stationary Poisson modeling provides an aggregate screening baseline, but must be paired with cycle-life tracking in real BMS deployments.
4. **Timeline Limitations:**
   - Active discharge cycles are stitched end-to-end, omitting charge protocols and open-circuit rest periods.
   - Cycle durations decrease from 3,690 seconds (1.02 h) to 2,792 seconds (0.78 h) as capacity fades.

---

## 🛡️ Engineering Decision Matrix

| Risk Classification | Decision Criteria ($P(N \ge k_{\text{crit}})$) | Recommended BMS Operational Protocol |
| :--- | :--- | :--- |
| 🟢 **LOW RISK** | $P < P_{\text{warn}}$ (e.g. $< 15\%$) | Standard nominal operation. Standard telemetry logging (0.1 Hz) and passive or baseline coolant flow loop active. |
| 🟡 **MODERATE RISK** | $P_{\text{warn}} \le P < P_{\text{high}}$ (e.g. $15\% - 40\%$) | Increase thermistor & cell voltage telemetry to 1 Hz. Pre-condition active liquid cooling circuit. Schedule module for automated internal resistance / EIS check at next cycle turnaround. |
| 🔴 **HIGH RISK** | $P \ge P_{\text{high}}$ (e.g. $\ge 40\%$) | **Immediate power curtailment:** Derate maximum charge/discharge C-rate by 50–70%. Command active chiller pump to maximum flow (100% duty cycle). Switch telemetry to high-speed transient logging (50–100 ms). Flag module for physical teardown if rate persists. |

---

## ⚠️ Important Engineering Screening Disclaimer

> **Statistical Screening Model Notice:**  
> This application implements a statistical screening model based on Poisson counting processes and Exponential waiting times.  
> **It is NOT a validated multi-physics battery safety or thermal runaway simulator.**  
> It does **not** solve 3D convective/conductive heat transfer (CFD), electrochemical heat generation ($q = I(V_{\text{oc}} - V) - I T \frac{\partial U}{\partial T}$), Arrhenius-type exothermic decomposition of the Solid Electrolyte Interphase (SEI), or cascading cell-to-cell thermal propagation. Physical battery qualification mandates multi-physics CAE models and standardized hardware qualification testing (e.g., UL 9540A, UN 38.3, SAE J2464).
