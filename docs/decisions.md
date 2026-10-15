# Decision log

| Date | Decision | Alternatives considered | Evidence (notebook / CV score) | Who |
|---|---|---|---|---|
| 2026-10-15 | Fill numeric gaps with iterative imputation (standardised columns), categorical gaps with an own level, inside each fold | median, mean, zero, KNN (k=5); mode, proportional (Week 3 plans) | `preprocessing_nico` §2.3: +0.0010 ± 0.0004 ROC AUC vs median / own level, 3/3 folds; all plans within ±0.0015 | Nico |
| 2026-10-15 | No missing-value flags for the history depth | 3 flags for 0 / 1 / 2 prior purchases | §2.4: −0.0002 ± 0.0008 ROC AUC, inside the noise | Nico |
| 2026-10-15 | No row deleted for outliers; the 1.5×IQR rule only screens | delete IQR outliers | §3.1: flags 66% of train rows, whose repeat rate is 0.60 vs 0.48 | Nico |
| 2026-10-15 | log1p on `days_since_previous_purchase`, `prior_mean_gap_days`, `prior_std_gap_days` only | none, all skewed, by group, winsorize 1/99 | §3.2: +0.0008 ± 0.0003 ROC AUC, 3/3 folds; winsorize −0.0031 (0/3); log1p on counts −0.0048 (0/3) | Nico |
| 2026-10-15 | Count encoding + standard scaling | 16 encoder × scaler combinations (Week 4) | §4: none better beyond 2 SEM; count + standard +0.0059 ± 0.0048 (3/3), and one-hot encodes 209 unseen `commercial_zone` test rows as the reference level | Nico |
| 2026-10-15 | Final recipe in `src/preprocessing.py` (`final_recipe()`) | V01 pipeline | §5: +0.0106 ± 0.0029 ROC AUC, 3/3 folds; honest F1 0.787 vs 0.771 | Nico |
