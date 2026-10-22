# CrowdSentinel experiment results

Synthetic rally ground, 4,000 people, 5 scenario seeds (0, 1, 2, 3, 4). Values are mean ± std over seeds. Density error = mean absolute error in persons/m² per zone, averaged over the whole event. Ground truth comes from the crowd simulation; tower and WiFi feeds are simulated (see README).

## Headline numbers

- Full system vs CCTV-only dashboard: density error reduced by **73%** (0.280 → 0.077 persons/m²).
- In zones with no camera: error reduced by **76%**.
- Active drone guidance vs fixed patrol: error in uncovered zones reduced by **49%**.
- Dangerous zones (≥ 4 persons/m²) caught by the upper estimate: 19% (CCTV only) vs **87%** (full system).
- Zones that truly turned critical: 25 across all seeds. Detected by the CCTV-only dashboard: 3. Detected by the full twin: **21** (threshold alarm), **25** with prediction, mean lead time 69 ± 14 s at 41% alert precision.
- Mean processing time per tick: 25.8 ms on this machine (the live loop runs every 2 s).

## 1. Fusion accuracy

| Configuration | Error, all zones | Error, no-camera zones | Error, occupied zones | Headcount error |
| --- | --- | --- | --- | --- |
| CCTV only (dashboard) | 0.280 ± 0.001 | 0.348 ± 0.000 | 2.178 ± 0.009 | 63.9% ± 1.2 |
| Weighted fusion, all sources | 0.307 ± 0.004 | 0.375 ± 0.002 | 1.416 ± 0.016 | 17.2% ± 0.8 |
| Kalman, CCTV only | 0.264 ± 0.002 | 0.323 ± 0.001 | 1.839 ± 0.018 | 47.1% ± 2.0 |
| Kalman, all sources, patrol drone | 0.139 ± 0.001 | 0.163 ± 0.002 | 0.624 ± 0.008 | 12.9% ± 0.5 |
| Kalman, all sources, ACTIVE drone | 0.077 ± 0.003 | 0.083 ± 0.003 | 0.306 ± 0.012 | 8.1% ± 1.1 |

## 2. Source ablation (full system minus one source)

| Configuration | Error, all zones | Error, no-camera zones | Headcount error |
| --- | --- | --- | --- |
| Kalman, all sources, ACTIVE drone | 0.077 ± 0.003 | 0.083 ± 0.003 | 8.1% ± 1.1 |
| - without drone | 0.278 ± 0.002 | 0.343 ± 0.001 | 13.7% ± 1.2 |
| - without cell towers | 0.070 ± 0.002 | 0.074 ± 0.002 | 5.9% ± 0.4 |
| - without WiFi | 0.077 ± 0.002 | 0.084 ± 0.002 | 9.4% ± 0.6 |
| - without CCTV | 0.105 ± 0.002 | 0.089 ± 0.001 | 13.1% ± 0.2 |

## 3. Drone guidance (core contribution)

| Drone policy | Error, no-camera zones | Dangerous zones caught |
| --- | --- | --- |
| Hover at centre | 0.326 ± 0.002 | 37.7 ± 0.8% |
| Random waypoints | 0.113 ± 0.010 | 86.7 ± 1.0% |
| Fixed patrol | 0.163 ± 0.002 | 68.8 ± 1.0% |
| **Active (uncertainty × risk)** | 0.083 ± 0.003 | 86.5 ± 0.9% |

## 4. Early warning (critical = 6 persons/m²)

Detected = an alarm fired for that zone between 5 min before and 1 min after it truly became critical. Lead time = seconds from the first alarm to the true crossing (positive = early). Precision = share of alarm episodes after which the zone really became critical within 5 min.

| Configuration | Alarm type | Detected / turned critical | Mean lead time (s) | Alarm precision |
| --- | --- | --- | --- | --- |
| CCTV only (dashboard) | threshold | 3 / 25 | 5 ± 4 | 83 ± 20% |
| CCTV only (dashboard) | predictive | 5 / 25 | 22 ± 7 | 39 ± 17% |
| Kalman, all sources, patrol drone | threshold | 10 / 25 | -6 ± 12 | 58 ± 6% |
| Kalman, all sources, patrol drone | predictive | 24 / 25 | -1 ± 7 | 22 ± 3% |
| Kalman, all sources, ACTIVE drone | threshold | 21 / 25 | -2 ± 11 | 100 ± 0% |
| Kalman, all sources, ACTIVE drone | predictive | 25 / 25 | 69 ± 14 | 41 ± 2% |

## Honest notes

- Ground truth, tower and WiFi data are simulated. Vision accuracy on real footage is measured separately (vision/evaluate_density.py on ShanghaiTech).
- The camera layout deliberately leaves the front-left of the stage uncovered, as real venues have blind spots. Results depend on this layout.
- Prediction is a smoothed-trend extrapolation, so lead times are short when a surge builds fast. Its settings were tuned on seed 0, which is also one of the evaluation seeds.
- Cell towers did not help in this layout: removing them slightly lowers zone error. Their 10 s delay and coarse sectors add noise when cameras and the drone already cover the crowd. They matter most when cameras fail (see the "without CCTV" row).
