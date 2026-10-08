# Role 2 - Poisson lead: step-by-step working

Data: `data/B0005_thermal_events.csv`, event = reading with temperature > 40 C,
X = number of event readings in one discharge cycle, window = cycles 400-614
(n = 56 cycles). Understand every line - you must be able to redo it in the viva.

## 1. Frequency table (count the cycles for each X)
| x | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 22 | 23 | total |
|---|----|----|----|----|----|----|----|----|----|----|----|----|-------|
| f | 1 | 1 | 4 | 14 | 1 | 7 | 8 | 7 | 2 | 8 | 1 | 2 | 56 |

## 2. Estimate lambda
Sum of x = sum(f * x) = 926.   lambda_hat = 926 / 56 = **16.5357** event readings per cycle
(Reason: for a Poisson variable E[X] = lambda, so the sample mean estimates lambda.)

## 3. Variance and dispersion check
Sum of x^2 = sum(f * x^2) = 15,746
s^2 = [ sum x^2 - (sum x)^2 / n ] / (n - 1) = [15,746 - 926^2/56] / 55
    = (15,746 - 15,312.07) / 55 = 433.93 / 55 = **7.8896**
Dispersion = s^2 / mean = 7.8896 / 16.5357 = **0.477**
A Poisson variable has variance = mean (ratio about 1). 0.477 means the counts are more
regular than Poisson (underdispersed): a limitation to state.

## 4. Poisson probabilities by recursion (lambda = 16.5357)
P(0) = e^(-16.5357) = 6.586 x 10^-8          P(k) = P(k-1) * lambda / k
| k | P(X = k) | k | P(X = k) |
|---|----------|---|----------|
| 10 | 0.0277 | 17 | 0.0957 |
| 12 | 0.0575 | 18 | 0.0879 |
| 14 | 0.0863 | 20 | 0.0632 |
| 15 | 0.0952 | 22 | 0.0374 |
| 16 | 0.0984 | 23 | 0.0269 |

## 5. Tail probabilities (use for the decision rule)
- P(X >= 3) = 1 - [P(0)+P(1)+P(2)] = 0.99999 (about 1, so useless as an alert rule here)
- P(X >= 10) = 0.9669
- P(X >= 15) = 0.6805
- **P(X >= 20) = 0.2270**   (a cycle with 20 or more event readings is unusual)

## 6. Observed vs expected (n = 56; E = 56 * P(X = k); merge groups until E >= 5)
| k | Observed O | Expected E | (O-E)^2/E |
|---|-----------|-----------|-----------|
| 0-11 | 1 | 5.74 | 3.918 |
| 12-13 | 5 | 7.31 | 0.730 |
| 14-15 | 15 | 10.16 | 2.302 |
| 16 | 7 | 5.51 | 0.404 |
| 17 | 8 | 5.36 | 1.304 |
| 18-19 | 9 | 9.20 | 0.005 |
| 20-21 | 8 | 6.33 | 0.441 |
| >= 22 | 3 | 6.38 | 1.793 |
| total | 56 | 56 | **10.896** |

dof = 8 categories - 1 - 1 (lambda was estimated) = 6. Critical value at 5% = 12.592.
10.896 < 12.592, so the Poisson fit is **not rejected** (p = 0.092), but it is borderline.

## 7. Interpretation
- The fit passes only narrowly. 14 cycles have exactly 14 readings (Poisson expects 4.8),
  and the spread is narrower than Poisson.
- Over ALL 168 cycles the fit is badly rejected (dispersion 8.6) because the event rate
  rises with battery age. Poisson is valid only within a stable-rate window.
- Model assumptions: independent events, constant rate, counted in equal intervals. They
  hold only approximately here (readings in a cycle are consecutive).

## 8. Likely viva questions for Role 2
1. State the Poisson formula and what each symbol means.
2. Why is lambda_hat the sample mean?
3. What does dispersion = 0.477 tell you?
4. Why merge categories before the chi-square test? Why dof = 6?
5. Why did you not use P(X >= 3) as the alert rule?
6. Why does the all-cycles data fail but cycles >= 400 pass?
