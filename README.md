# EV Battery Thermal Events: Poisson and Exponential Models

**Course:** Statistical Methods and Probability, SY BTech | **IA-1:** STAT-AI Engineering Challenge
**Group 11 | Topic 11:** Poisson and Exponential Models for Battery Events

## Problem
How often do abnormal thermal events occur, and how long until the next one? We model
event counts per week with a Poisson distribution and the waiting time between events
with an exponential distribution, then use the result for a battery-management alert.

## Data
- **Main dataset:** `data/B0005_thermal_events.csv` (group's own file): 50,285 temperature
  readings over 168 battery discharge cycles; `thermal_event` = reading above 40 C.
  Needs faculty approval as a source (see `docs/dataset_findings_B0005.md`).
- Poisson variable: event readings per discharge cycle. Exponential variable: time from
  start of discharge to the first event reading.
- The app and CLI also accept daily NASA POWER files (`python -m src.fetch_data`).
- What the data actually shows (dispersion, trend, fit tests): `docs/dataset_findings_B0005.md`

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.run_analysis data/B0005_thermal_events.csv --threshold 40 --cycle-min 400 --k 10 --p 0.5
streamlit run app.py                      # live demo for the showcase
pytest                                    # checks code against the hand calculations
```

## Demo flow (matches the IA requirement)
`Input/Data -> Statistical Method -> Result -> Engineering/AI Decision`

## Repository layout
| Path | Purpose |
|------|---------|
| `app.py` | Streamlit interactive demo |
| `src/stats_core.py` | Poisson, exponential, chi-square, decision rule (plain `math`, easy to explain) |
| `src/data_prep.py` | Weekly counts and event gaps from daily data |
| `src/fetch_data.py` | NASA POWER download + metadata for the citation |
| `src/run_analysis.py` | Command-line run that saves figures to `outputs/` |
| `src/simulate_data.py` | Offline test data only (clearly marked SIMULATED) |
| `tests/` | Unit tests that reproduce the hand-calculated example |
| `docs/` | Worked example template, investigation sheet outline, viva prep |

## Team (fill in)
| Member | Contribution |
|--------|--------------|
| 1 | Dataset and assumptions |
| 2 | Hand calculations: Poisson |
| 3 | Hand calculations: exponential |
| 4 | Python verification (`src/`, `tests/`) |
| 5 | Streamlit demo (`app.py`) |
| 6 | Graphs, investigation sheet, references |

## Limitations
Event readings within a cycle are consecutive (not independent); the event rate grows with
battery age; one cell only; the 40 C flag is chosen by the group.

## References
Original source of B0005_thermal_events.csv: add after confirming with your faculty.
Optional ambient-data alternative: NASA POWER Project, https://power.larc.nasa.gov/
