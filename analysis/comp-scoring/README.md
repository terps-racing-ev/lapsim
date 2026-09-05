# FSAE Michigan EV competition scoring

This directory contains reproducible Formula SAE Michigan EV scoring plots from
2023 through 2026. The cumulative plot shows the official 9th- and 10th-place
score cutoffs. Each individual-event plot shows the median score among the
5th-through-10th-place teams. The median is used instead of a geometric mean
because it is robust to outliers and remains meaningful when an event score is
zero. Blank event cells in the official results are represented as zero.
Official displayed totals are used directly, so they can differ from the sum
of rounded component scores by 0.1 point.

The Maryland 2026 actual references come from its official 364.8-point result.
The projected endurance-plus-efficiency result defaults to 254.4 points. The
projected cumulative total replaces Maryland's actual 6.0 endurance-plus-
efficiency points, producing 613.2 points.

Run:

```powershell
python analysis/comp-scoring/plot_michigan_ev_scores.py
```

Outputs:

- `cumulative_scores.png`
- `cumulative_scores_no_maryland.png`
- `cumulative_scores_projected_only.png`
- `static_event_scores.png`
- `static_event_scores_no_maryland.png`
- `acceleration_skidpad_scores.png`
- `acceleration_skidpad_scores_no_maryland.png`
- `autocross_endurance_efficiency_scores.png`
- `autocross_endurance_efficiency_scores_no_maryland.png`

Data sources:

- [2023 official results](https://brx-content.fullsight.org/site/binaries/content/assets/sae-org/content/events/student/about-formula/sae-electric/formula-sae-electric/fsae_ev_2023_results.pdf)
- [2024 official results](https://brx-content.fullsight.org/site/binaries/content/assets/sae-org/content/events/student/about-formula/sae-electric/formula-sae-electric/fsae_ev_2024_results.pdf)
- [2025 official results](https://www.fsaeonline.com/CompResources/2025/8f030a58-d9e4-49b8-bc83-6ca16c7ce715/FSAE_2025_MI6_results.pdf)
- [2026 official results](https://www.fsaeonline.com/CompResources/2026/07af50d8-cbb6-4b9b-aaf8-5ff6a7e44057/FSAE_2026_MI6_results.pdf)
