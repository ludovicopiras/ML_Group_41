"""Learned preprocessing steps, built as scikit-learn transformers / Pipelines.

This module will hold the preprocessing chosen in notebooks/lab/04_preprocessing.ipynb:
imputation (e.g. median, KNN, missing-value indicators), categorical encoding
(one-hot, ordinal, target encoding for high-cardinality columns), scaling and any
learned outlier thresholds, assembled with ColumnTransformer / Pipeline.

Everything here learns from data, so it must always be fitted on training folds
only (inside a Pipeline passed to cross-validation) and never on test.csv.
"""
