# EV 2026 autocross score correlation

This directory contains a small linear ratio model that predicts
Formula SAE Electric 2026 autocross score from acceleration and skidpad score.
The source is the `Formula SAE Electric 2026 Overall Results` table in
`FSAE_2026_MI6_results.pdf`, pages 1-4.

The fitted model is:

```text
predicted_autocross = clip(
    125 * (
        24.636470 / 125
        + 0.133094 * acceleration_score / 100
        + 0.743173 * skidpad_score / 75
    ),
    0,
    125,
)
```

The weights and y-intercept are fitted by least squares in normalized score
ratios. Blank event scores are excluded; teams with an explicit `0.0`
autocross score are excluded from this fit.

Run from the repository root:

```powershell
python analysis\autox_corr\fit_model.py
```

The script writes `outputs/model_summary.json`, `outputs/predictions.csv`, and
`outputs/autocross_prediction.png`. The plot highlights the top 10 observed
autocross scores and Univ. of Maryland separately. The generated summary
contains the exact weights and fit metrics; the rounded equation above is only
for readability.

This is a descriptive baseline, not a vehicle-performance model. The fit uses
31 teams with complete scores after removing zero-autocross results, and the
data include event penalties, DNFs, and other competition effects already
reflected in the published points. The
in-sample RMSE should therefore be read as a rough error scale, not as a
guaranteed prediction interval for another event.
