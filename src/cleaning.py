"""Deterministic, rule-based data cleaning.

This module will hold the cleaning rules developed in notebooks/lab/02_cleaning.ipynb,
for example: fixing column types, renaming Portuguese column names, turning
impossible values (e.g. negative quantities or totals that cannot exist) into NaN,
and harmonising inconsistent category labels.

Rules here must NOT learn anything from the data (no means, medians, quantiles),
so they can be applied identically to train.csv and test.csv.
Rows are never dropped from the test set.
"""
