# Decision log

One row per non-trivial decision (cleaning rule, validation scheme, preprocessing choice, feature kept/dropped,
model, hyperparameters, threshold). The final notebook summarises these decisions, and they are what we
defend in the discussion, so always write down the alternatives and the evidence.

- **Date**: `YYYY-MM-DD`
- **Evidence**: notebook + section, and the CV score (mean ± std F1) when relevant
- **Who**: initials or GitHub username

| Date | Decision | Alternatives considered | Evidence (notebook / CV score) | Who |
|---|---|---|---|---|
| _EXAMPLE — 2026-10-10_ | _EXAMPLE: drop `random_noise` from the feature set_ | _Keep it; keep only if selected by RFE_ | _EXAMPLE: `05_features` §1 — no relationship with target; CV F1 0.712 ± 0.015 without vs 0.709 ± 0.016 with_ | _XX_ |
