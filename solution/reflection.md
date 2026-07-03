# Reflection (≤1 page)

**Which fault types were hardest to catch, and why?**

Private-phase subtle `data_batch` faults were hardest: `distribution_shift` and
`freshness_lag` with metrics squarely inside the published 3σ baselines (e.g.
row=508, mean=85.7, stale=6.1). `runtime_anomaly`, `embedding_drift`, and
`corpus_staleness` also sit near healthy variance. A single global threshold
cannot separate them; catching most required per-metric sliding-window z-scores
on stream self-history (Welford), with combo=1 so one strong signal suffices.

**What would you change about your cost/coverage tradeoff, if you had another pass?**

The scoring formula caps at **50** with perfect TPR/FPR and zero cost overage.
This defense accepts extra false positives to preserve recall on subtle private
faults: practice reached TPR 1.0 with FPR around 18%, while private finished
near **42** with TPR around 96% and FPR around 21%.

I kept one metered tool call per event (private budget 320 covers ~300 credits).
With another pass I would add modified z-scores (MAD-based, per NIST/Iglewicz) as
a robust secondary layer and multivariate joint rules (mean↑ + std↓ for
distribution shift), tuning on practice labels while accepting that practice=50
and private≈42 are on a Pareto frontier with a single static defense.
