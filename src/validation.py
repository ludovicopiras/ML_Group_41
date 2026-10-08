"""Validation scheme for model assessment.

This module will hold the splitting strategy designed in notebooks/lab/03_validation.ipynb:
a time-aware and/or customer-aware cross-validation (e.g. splits ordered by
purchase_date, grouped by customer_id) that mimics the relation between train.csv
and the chronologically later test.csv, plus helpers to score models with F1
(positive class) and to tune the decision threshold on out-of-fold predictions.
"""
